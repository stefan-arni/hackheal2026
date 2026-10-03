//
//  CameraManager.swift
//  NoseThumb
//

import AVFoundation
import Observation

// Which camera is in use. Both deliver video plus a depth map.
nonisolated enum CameraPosition: Sendable {
    case front   // TrueDepth (Face ID) camera
    case back    // LiDAR camera (Pro iPhones)
}

// Owns the camera session. All session work runs on `sessionQueue`,
// never on the main thread, so the UI stays smooth.
// Each video frame arrives together with a depth map (distance from the phone,
// per pixel) on `videoQueue`.
@Observable
nonisolated final class CameraManager: NSObject, AVCaptureDataOutputSynchronizerDelegate, @unchecked Sendable {
    // Frames per second, shown on screen. Only changed on the main thread.
    private(set) var fps = 0
    private(set) var depthFPS = 0   // how many of those frames came with a depth map
    private(set) var validDepthPercent = 0   // temporary debug: share of depth pixels with a valid value
    private(set) var position: CameraPosition = .front
    // Temporary debug text shown at the bottom of the screen.
    private(set) var debugText = "DEBUG: waiting for depth…"

    @ObservationIgnored let poseDetector = PoseDetector()
    @ObservationIgnored let session = AVCaptureSession()
    @ObservationIgnored private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    @ObservationIgnored private let videoQueue = DispatchQueue(label: "camera.video.queue")

    @ObservationIgnored private let videoOutput = AVCaptureVideoDataOutput()
    @ObservationIgnored private let depthOutput = AVCaptureDepthDataOutput()
    // Delivers video and depth from the same moment together.
    @ObservationIgnored private var synchronizer: AVCaptureDataOutputSynchronizer?

    // Used only on `videoQueue` to compute fps.
    @ObservationIgnored private var frameCount = 0
    @ObservationIgnored private var depthFrameCount = 0
    @ObservationIgnored private var lastDepthData: AVDepthData?
    @ObservationIgnored private var windowStart = ProcessInfo.processInfo.systemUptime

    // Temporary debug: print video vs depth sizes once per camera.
    @ObservationIgnored private var printedDebugInfo = false
    @ObservationIgnored private var cameraText = "camera: not set"
    @ObservationIgnored private var depthFormatText = "depth format: not set"
    @ObservationIgnored private var fieldOfViewText = "field of view: not set"

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else {
                print("Camera permission denied")
                return
            }
            self.sessionQueue.async {
                // Already set up (e.g. the view appeared twice): nothing to do.
                guard self.session.inputs.isEmpty else { return }
                self.configureAndRun(.front)
            }
        }
    }

    // Switches between the front and back camera. Called from the UI (main thread).
    func switchCamera() {
        let newPosition: CameraPosition = position == .front ? .back : .front
        position = newPosition
        sessionQueue.async {
            self.configureAndRun(newPosition)
        }
    }

    private func configureAndRun(_ position: CameraPosition) {
        // Start from a clean session: stop, remove the old camera and outputs.
        if session.isRunning {
            session.stopRunning()
        }
        session.beginConfiguration()
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        synchronizer = nil

        let device: AVCaptureDevice?
        switch position {
        case .front:
            // Apple's TrueDepth sample uses this size; it supports depth.
            session.sessionPreset = .vga640x480
            device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front)
            cameraText = "camera: front TrueDepth"
        case .back:
            device = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
            cameraText = "camera: back LiDAR"
        }

        guard let device,
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            print("Could not set up the \(position) camera")
            showDebug("Could not set up the \(position) camera")
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        // The LiDAR camera has no size preset with depth: pick a video format from its list.
        // Use the smallest one (at least 640 wide) that supports depth, so MediaPipe stays fast.
        if position == .back {
            let withDepth = device.formats.filter {
                !$0.supportedDepthDataFormats.isEmpty && !$0.isVideoBinned &&
                CMVideoFormatDescriptionGetDimensions($0.formatDescription).width >= 640
            }
            guard let format = withDepth.min(by: {
                CMVideoFormatDescriptionGetDimensions($0.formatDescription).width <
                CMVideoFormatDescriptionGetDimensions($1.formatDescription).width
            }) else {
                print("No LiDAR video format with depth")
                showDebug("No LiDAR video format with depth")
                session.commitConfiguration()
                return
            }
            do {
                try device.lockForConfiguration()
                device.activeFormat = format
                device.unlockForConfiguration()
            } catch {
                print("Could not set LiDAR video format: \(error)")
            }
        }

        // Output 1: video frames (for MediaPipe).
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        guard session.canAddOutput(videoOutput) else {
            print("Could not add video output")
            session.commitConfiguration()
            return
        }
        session.addOutput(videoOutput)

        // Output 2: depth maps.
        // Front: filtering off = raw measurements (holes are NaN, we skip them).
        // Back (LiDAR): the scanner only measures a sparse grid of points; without filtering
        // many pixels (e.g. the face) stay empty, so let the system fill the holes.
        depthOutput.isFilteringEnabled = position == .back
        depthOutput.alwaysDiscardsLateDepthData = true
        guard session.canAddOutput(depthOutput) else {
            print("Could not add depth output")
            session.commitConfiguration()
            return
        }
        session.addOutput(depthOutput)

        // Rotate both streams upright (portrait). Mirror only the front camera (selfie view),
        // matching the on-screen preview.
        for connection in [videoOutput.connection(with: .video), depthOutput.connection(with: .depthData)] {
            guard let connection else { continue }
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = position == .front
            }
        }

        // Pick a depth format measured in meters (Float32 if offered, else Float16), largest size.
        let depthFormats = device.activeFormat.supportedDepthDataFormats
        let float32 = depthFormats.filter { CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32 }
        let float16 = depthFormats.filter { CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16 }
        let candidates = float32.isEmpty ? float16 : float32
        if let best = candidates.max(by: {
            CMVideoFormatDescriptionGetDimensions($0.formatDescription).width <
            CMVideoFormatDescriptionGetDimensions($1.formatDescription).width
        }) {
            do {
                try device.lockForConfiguration()
                device.activeDepthDataFormat = best
                device.unlockForConfiguration()
                let size = CMVideoFormatDescriptionGetDimensions(best.formatDescription)
                print("Depth format: \(size.width)x\(size.height)")
                depthFormatText = "depth format: \(size.width)x\(size.height)"
            } catch {
                print("Could not set depth format: \(error)")
            }
        } else {
            print("No depth formats available for this camera format")
            depthFormatText = "depth format: none"
        }

        // Needed to turn pixels into cm.
        let fieldOfView = device.activeFormat.videoFieldOfView
        poseDetector.setFieldOfView(fieldOfView)
        fieldOfViewText = String(format: "field of view: %.1f°", fieldOfView)

        let synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer.setDelegate(self, queue: videoQueue)
        self.synchronizer = synchronizer

        session.commitConfiguration()

        // Show the debug info again for the new camera.
        videoQueue.async {
            self.printedDebugInfo = false
        }
        showDebug("\(cameraText)\nwaiting for depth…")

        session.startRunning()
    }

    private func showDebug(_ text: String) {
        DispatchQueue.main.async {
            self.debugText = text
        }
    }

    // Called on `videoQueue` with a video frame and the depth map from the same moment.
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        guard let syncedVideo = synchronizedDataCollection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
              !syncedVideo.sampleBufferWasDropped else { return }

        // Depth may be missing for some frames; still run pose detection without it.
        var depthData: AVDepthData?
        if let syncedDepth = synchronizedDataCollection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData,
           !syncedDepth.depthDataWasDropped {
            depthData = syncedDepth.depthData
        }

        // Temporary debug: are video and depth the same shape and orientation?
        if !printedDebugInfo, let depthData, let videoBuffer = CMSampleBufferGetImageBuffer(syncedVideo.sampleBuffer) {
            printedDebugInfo = true
            let depthMap = depthData.depthDataMap
            var lines = [cameraText, depthFormatText, fieldOfViewText]
            lines.append("video frame: \(CVPixelBufferGetWidth(videoBuffer))x\(CVPixelBufferGetHeight(videoBuffer))")
            lines.append("depth map: \(CVPixelBufferGetWidth(depthMap))x\(CVPixelBufferGetHeight(depthMap))")
            for (name, connection) in [("video", videoOutput.connection(with: .video)),
                                       ("depth", depthOutput.connection(with: .depthData))] {
                if let connection {
                    lines.append("\(name): rot \(Int(connection.videoRotationAngle)) mirror \(connection.isVideoMirrored) | rot90 ok \(connection.isVideoRotationAngleSupported(90)) mirror ok \(connection.isVideoMirroringSupported)")
                } else {
                    lines.append("\(name): no connection")
                }
            }
            let text = lines.joined(separator: "\n")
            print(text)
            DispatchQueue.main.async {
                self.debugText = text
            }
        }

        poseDetector.detect(sampleBuffer: syncedVideo.sampleBuffer, depthData: depthData)

        frameCount += 1
        if depthData != nil {
            depthFrameCount += 1
            lastDepthData = depthData
        }

        let now = ProcessInfo.processInfo.systemUptime
        let elapsed = now - windowStart
        if elapsed >= 1 {
            let newFPS = Int((Double(frameCount) / elapsed).rounded())
            let newDepthFPS = Int((Double(depthFrameCount) / elapsed).rounded())
            // Once a second: how much of the latest depth map has valid values.
            let newValidPercent = lastDepthData
                .flatMap { PoseDetector.validDepthFraction($0) }
                .map { Int(($0 * 100).rounded()) } ?? 0
            frameCount = 0
            depthFrameCount = 0
            windowStart = now
            DispatchQueue.main.async {
                self.fps = newFPS
                self.depthFPS = newDepthFPS
                self.validDepthPercent = newValidPercent
            }
        }
    }
}
