// DepthCamera.swift
// posecam iPhone side: captures video + depth together and hands each frame over
// ready to send to the pose server.
//
//   Back camera:  LiDAR depth (iPhone 12 Pro and later Pro models). Works at
//                 full-body distance (2-3 m); use this for the sway tests.
//   Front camera: TrueDepth (Face ID iPhones). Accurate only up to ~1 m, so at
//                 full-body distance the depth is mostly missing or noisy; the
//                 server then falls back to 2D (side-to-side sway only).
//
// Each frame: a JPEG (640 px on the long side), the depth map as little-endian
// UInt16 millimetres (<= 320 px wide), the camera intrinsics scaled to the depth
// map, the clockwise rotation that makes it upright, and the capture timestamp.
// Video and depth come from one AVCaptureDataOutputSynchronizer with a 4:3
// format, so they show the same field of view (the server relies on that).
//
// Requires iOS 17+. Info.plist: NSCameraUsageDescription.

import AVFoundation
import CoreImage
import Foundation
import ImageIO
import SwiftUI
import UIKit

nonisolated enum DepthCameraKind: String, CaseIterable, Identifiable, Sendable {
    case lidar, truedepth

    var id: String { rawValue }

    var title: String {
        switch self {
        case .lidar: return "Back · LiDAR"
        case .truedepth: return "Front · TrueDepth"
        }
    }

    var deviceType: AVCaptureDevice.DeviceType {
        self == .lidar ? .builtInLiDARDepthCamera : .builtInTrueDepthCamera
    }

    var position: AVCaptureDevice.Position { self == .lidar ? .back : .front }

    var isAvailable: Bool {
        AVCaptureDevice.default(deviceType, for: .video, position: position) != nil
    }
}

nonisolated struct DepthFrame: Sendable {
    let index: Int
    let jpeg: Data
    let depthMillimetres: Data?     // UInt16 little-endian, row-major, 0 = no depth
    let depthWidth: Int
    let depthHeight: Int
    let intrinsics: [Double]        // fx, fy, cx, cy in depth-map pixels
    let rotate: Int                 // clockwise degrees to make the frame upright
    let timestampMs: Double
    let camera: String
}

nonisolated struct CameraError: LocalizedError {
    let errorDescription: String?
    init(_ text: String) { errorDescription = text }
}

/// Runs the capture session on its own queue. Not tied to the main actor.
nonisolated final class CameraEngine: NSObject, AVCaptureDataOutputSynchronizerDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    var onFrame: (@Sendable (DepthFrame) -> Void)?
    var onState: (@Sendable (_ running: Bool, _ error: String?) -> Void)?
    var maxFPS: Double = 30
    var jpegLongSide: CGFloat = 640
    var maxDepthWidth = 320

    private let queue = DispatchQueue(label: "posecam.camera")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private var device: AVCaptureDevice?
    private var kind: DepthCameraKind = .lidar
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var lastSent = -Double.infinity
    private var frameIndex = 0

    func start(_ kind: DepthCameraKind) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else {
                self.onState?(false, "Camera access is off. Allow it in Settings > Privacy & Security > Camera.")
                return
            }
            self.queue.async { self.configureAndRun(kind) }
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
            self.onState?(false, nil)
        }
    }

    private func configureAndRun(_ kind: DepthCameraKind) {
        if session.isRunning { session.stopRunning() }
        session.beginConfiguration()
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        synchronizer = nil
        do {
            guard let dev = AVCaptureDevice.default(kind.deviceType, for: .video, position: kind.position) else {
                throw CameraError(kind == .lidar
                    ? "This iPhone has no LiDAR camera (it's on iPhone 12 Pro and later Pro models)."
                    : "This iPhone has no TrueDepth camera.")
            }
            session.sessionPreset = .inputPriority
            let input = try AVCaptureDeviceInput(device: dev)
            guard session.canAddInput(input) else { throw CameraError("Can't use this camera.") }
            session.addInput(input)

            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            guard session.canAddOutput(videoOutput) else { throw CameraError("Can't add video output.") }
            session.addOutput(videoOutput)
            depthOutput.isFilteringEnabled = true          // fills small holes, smooths edges
            depthOutput.alwaysDiscardsLateDepthData = true
            guard session.canAddOutput(depthOutput) else { throw CameraError("Can't add depth output.") }
            session.addOutput(depthOutput)

            // smallest 4:3 video format (>= 640 wide) that supports depth
            func width(_ f: AVCaptureDevice.Format) -> Int32 {
                CMVideoFormatDescriptionGetDimensions(f.formatDescription).width
            }
            let formats = dev.formats.filter { f in
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                return !f.supportedDepthDataFormats.isEmpty && d.width * 3 == d.height * 4 && d.width >= 640
            }.sorted { width($0) < width($1) }
            guard let format = formats.first else { throw CameraError("No 4:3 video format with depth on this camera.") }
            try dev.lockForConfiguration()
            dev.activeFormat = format
            let depthFormats = format.supportedDepthDataFormats.filter {
                let t = CMFormatDescriptionGetMediaSubType($0.formatDescription)
                return t == kCVPixelFormatType_DepthFloat16 || t == kCVPixelFormatType_DepthFloat32
            }
            if let best = depthFormats.max(by: { width($0) < width($1) }) {
                dev.activeDepthDataFormat = best
            }
            dev.unlockForConfiguration()

            if let c = videoOutput.connection(with: .video), c.isVideoMirroringSupported {
                c.automaticallyAdjustsVideoMirroring = false
                c.isVideoMirrored = false                    // server expects the camera's view
            }
            depthOutput.connection(with: .depthData)?.isEnabled = true

            device = dev
            self.kind = kind
            let sync = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
            sync.setDelegate(self, queue: queue)
            synchronizer = sync
            session.commitConfiguration()
            session.startRunning()
            onState?(session.isRunning, session.isRunning ? nil : "The camera didn't start.")
        } catch {
            session.commitConfiguration()
            onState?(false, error.localizedDescription)
        }
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let video = collection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
              !video.sampleBufferWasDropped,
              let pixels = CMSampleBufferGetImageBuffer(video.sampleBuffer) else { return }
        let t = CMTimeGetSeconds(video.timestamp)
        guard t - lastSent >= 1.0 / maxFPS - 0.002 else { return }
        lastSent = t

        var depth: (data: Data, w: Int, h: Int, k: [Double])?
        if let d = collection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData,
           !d.depthDataWasDropped {
            depth = encodeDepth(d.depthData)
        }
        guard let jpeg = encodeJPEG(pixels) else { return }
        // Buffers arrive in the sensor's landscape orientation. The app is
        // portrait-only, and 90 degrees clockwise is portrait for both cameras
        // (AVCaptureConnection.videoRotationAngle). The server rotates the image
        // and the depth map together.
        let rotate = 90
        frameIndex += 1
        onFrame?(DepthFrame(index: frameIndex, jpeg: jpeg, depthMillimetres: depth?.data,
                            depthWidth: depth?.w ?? 0, depthHeight: depth?.h ?? 0,
                            intrinsics: depth?.k ?? [], rotate: rotate,
                            timestampMs: t * 1000, camera: kind.rawValue))
    }

    private func encodeJPEG(_ pixels: CVPixelBuffer) -> Data? {
        var image = CIImage(cvPixelBuffer: pixels)
        let scale = jpegLongSide / max(image.extent.width, image.extent.height)
        if scale < 1 {
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        guard let srgb = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        return ciContext.jpegRepresentation(of: image, colorSpace: srgb, options: [quality: 0.7])
    }

    /// Depth map -> UInt16 millimetres (downsampled to <= maxDepthWidth) + intrinsics.
    private func encodeDepth(_ raw: AVDepthData) -> (data: Data, w: Int, h: Int, k: [Double])? {
        let data = raw.depthDataType == kCVPixelFormatType_DepthFloat32
            ? raw : raw.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = data.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let step = max(1, Int((Double(w) / Double(maxDepthWidth)).rounded(.up)))
        let ow = w / step, oh = h / step
        var out = [UInt16](repeating: 0, count: ow * oh)
        for y in 0..<oh {
            let row = base.advanced(by: y * step * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in 0..<ow {
                let z = row[x * step]                       // metres
                out[y * ow + x] = (z.isFinite && z > 0) ? UInt16(min(z * 1000, 65535)) : 0
            }
        }
        let bytes = out.withUnsafeBufferPointer { Data(buffer: $0) }   // iOS is little-endian

        var k: [Double]
        if let cal = data.cameraCalibrationData {
            let m = cal.intrinsicMatrix                     // column-major 3x3
            let ref = cal.intrinsicMatrixReferenceDimensions
            let sx = Double(w) / Double(ref.width) / Double(step)
            let sy = Double(h) / Double(ref.height) / Double(step)
            k = [Double(m.columns.0.x) * sx, Double(m.columns.1.y) * sy,
                 Double(m.columns.2.x) * sx, Double(m.columns.2.y) * sy]
        } else {
            // no calibration: estimate from the field of view (centre of the image)
            let fov = Double(device?.activeFormat.videoFieldOfView ?? 60) * .pi / 180
            let fx = Double(ow) / 2 / tan(fov / 2)
            k = [fx, fx, Double(ow - 1) / 2, Double(oh - 1) / 2]
        }
        return (bytes, ow, oh, k)
    }
}

/// SwiftUI-facing wrapper around CameraEngine.
@MainActor
final class DepthCamera: ObservableObject {
    @Published private(set) var running = false
    @Published private(set) var kind: DepthCameraKind = .lidar
    @Published var error: String?

    let engine = CameraEngine()
    var session: AVCaptureSession { engine.session }

    init() {
        engine.onState = { [weak self] running, error in
            Task { @MainActor [weak self] in
                self?.running = running
                self?.error = error
            }
        }
    }

    func start(_ kind: DepthCameraKind) {
        self.kind = kind
        error = nil
        engine.start(kind)
    }

    func stop() { engine.stop() }

    /// Called on the camera queue for every frame (up to 30 fps).
    func setFrameHandler(_ handler: @escaping @Sendable (DepthFrame) -> Void) {
        engine.onFrame = handler
    }
}

// MARK: - Camera preview + skeleton

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let running: Bool

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        var session: AVCaptureSession?

        // A camera session shows in only one preview at a time: attach it only
        // while this view is on screen.
        override func didMoveToWindow() {
            super.didMoveToWindow()
            previewLayer.session = window == nil ? nil : session
            updateRotation()
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            updateRotation()
        }

        func updateRotation() {
            // the app is portrait-only; the server rotates frames the same way
            if let c = previewLayer.connection, c.isVideoRotationAngleSupported(90) {
                c.videoRotationAngle = 90
            }
        }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.session = session
        v.previewLayer.videoGravity = .resizeAspect
        v.backgroundColor = .black
        return v
    }

    func updateUIView(_ v: PreviewView, context: Context) {
        v.updateRotation()
    }
}

struct SkeletonOverlay: View {
    let points: [Int: CGPoint]          // normalized, on the upright frame the server saw
    let mirrored: Bool                  // front-camera preview is mirrored

    static let bones: [(Int, Int)] = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15),
                                      (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
                                      (27, 31), (28, 32)]

    var body: some View {
        Canvas { ctx, size in
            func p(_ i: Int) -> CGPoint? {
                guard let q = points[i] else { return nil }
                return CGPoint(x: (mirrored ? 1 - q.x : q.x) * size.width, y: q.y * size.height)
            }
            var path = Path()
            for (a, b) in Self.bones {
                if let pa = p(a), let pb = p(b) {
                    path.move(to: pa)
                    path.addLine(to: pb)
                }
            }
            ctx.stroke(path, with: .color(.green), lineWidth: 3)
            for i in points.keys {
                if let q = p(i) {
                    ctx.fill(Path(ellipseIn: CGRect(x: q.x - 4, y: q.y - 4, width: 8, height: 8)),
                             with: .color(.yellow))
                }
            }
        }
        .allowsHitTesting(false)
    }
}
