import SwiftUI
import AVFoundation
import Combine
import UIKit
import WebRTC
import CoreMedia
import CoreVideo

struct ContentView: View {

    @StateObject private var camera = CameraManager()
    @StateObject private var streamer = VideoStreamClient()
    @StateObject private var webRTC = WebRTCClient()

    @State private var callRoom = "television-demo"
    @State private var callStarted = false

    @State private var activeTest: String?
    @State private var activeTestSessionID: String?

    @AppStorage("poseServerIP")
    private var serverIP = "10.0.6.237"

    private var connectionDotColor: Color {
        if webRTC.isConnected {
            return .green
        }

        if callStarted {
            return .orange
        }

        return .gray
    }

    private var connectionLabel: String {
        if webRTC.isConnected {
            return "Connected"
        }

        if callStarted {
            return "Connecting"
        }

        return "Offline"
    }

    private var waitingIconName: String {
        callStarted
            ? "person.crop.rectangle.badge.clock"
            : "video.fill"
    }

    private var waitingText: String {
        callStarted
            ? "Waiting for doctor..."
            : "Ready for appointment"
    }

    private var callButtonIcon: String {
        callStarted
            ? "phone.down.fill"
            : "video.fill"
    }

    private var callButtonText: String {
        callStarted
            ? "End Call"
            : "Join Doctor Call"
    }

    var body: some View {

        ZStack {

            Color.black
                .ignoresSafeArea()

            // =====================================================
            // DOCTOR VIDEO
            // =====================================================

            if let remoteTrack = webRTC.remoteVideoTrack {

                RemoteVideoView(
                    videoTrack: remoteTrack
                )
                .ignoresSafeArea()

            } else {

                VStack(spacing: 14) {

                    Image(
                        systemName: waitingIconName
                    )
                    .font(.system(size: 50))
                    .foregroundStyle(
                        .white.opacity(0.65)
                    )

                    Text(waitingText)
                        .font(.headline)
                        .foregroundStyle(
                            .white.opacity(0.8)
                        )

                    if callStarted {

                        Text(
                            webRTC.connectionStatus
                        )
                        .font(.caption)
                        .foregroundStyle(
                            .white.opacity(0.55)
                        )
                    }
                }
            }

            // =====================================================
            // CALL UI
            // =====================================================

            VStack {

                HStack {

                    VStack(
                        alignment: .leading,
                        spacing: 2
                    ) {

                        Text("TeleVision")
                            .font(.headline)
                            .foregroundStyle(.white)

                        if callStarted {

                            Text(
                                webRTC.connectionStatus
                            )
                            .font(.caption2)
                            .foregroundStyle(
                                .white.opacity(0.65)
                            )
                        }
                    }

                    Spacer()

                    HStack(spacing: 7) {

                        Circle()
                            .fill(
                                connectionDotColor
                            )
                            .frame(
                                width: 9,
                                height: 9
                            )

                        Text(
                            connectionLabel
                        )
                        .font(.caption.bold())
                        .foregroundStyle(.white)
                    }
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .background(
                        .black.opacity(0.55)
                    )
                    .clipShape(Capsule())
                }
                .padding(.horizontal)
                .padding(.top, 8)

                Spacer()

                // =================================================
                // PATIENT SELF VIEW
                // =================================================

                HStack {

                    Spacer()

                    VStack(
                        alignment: .trailing,
                        spacing: 5
                    ) {

                        CameraPreview(
                            session: camera.session
                        )
                        .frame(
                            width: 120,
                            height: 180
                        )
                        .background(Color.black)
                        .clipShape(
                            RoundedRectangle(
                                cornerRadius: 17
                            )
                        )
                        .overlay {

                            RoundedRectangle(
                                cornerRadius: 17
                            )
                            .stroke(
                                .white.opacity(0.45),
                                lineWidth: 1
                            )
                        }
                        .shadow(
                            color: .black.opacity(0.5),
                            radius: 8
                        )

                        Text("You")
                            .font(.caption2.bold())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                .black.opacity(0.55)
                            )
                            .clipShape(Capsule())
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 8)

                // =================================================
                // CONNECTION SETTINGS
                // =================================================

                if !callStarted {

                    VStack(spacing: 8) {

                        TextField(
                            "Mac IP",
                            text: $serverIP
                        )
                        .textFieldStyle(
                            .roundedBorder
                        )
                        .keyboardType(
                            .numbersAndPunctuation
                        )
                        .autocorrectionDisabled()

                        TextField(
                            "Room",
                            text: $callRoom
                        )
                        .textFieldStyle(
                            .roundedBorder
                        )
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(
                            .never
                        )
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 6)
                }

                // =================================================
                // CALL BUTTON
                // =================================================

                Button {

                    if callStarted {
                        endCall()
                    } else {
                        startCall()
                    }

                } label: {

                    HStack(spacing: 9) {

                        Image(
                            systemName: callButtonIcon
                        )

                        Text(
                            callButtonText
                        )
                        .fontWeight(
                            .semibold
                        )
                    }
                    .frame(
                        maxWidth: .infinity
                    )
                    .padding(
                        .vertical,
                        6
                    )
                }
                .buttonStyle(
                    .borderedProminent
                )
                .tint(
                    callStarted
                        ? Color.red
                        : Color.blue
                )
                .padding(.horizontal)
                .padding(.bottom, 12)
            }
        }

        // =========================================================
        // PATIENT INSTRUCTIONS DURING A BESS STANCE
        // (big, readable from 2-3 m; no scores or numbers)
        // =========================================================

        .overlay(alignment: .top) {
            let inStance = activeTest == "balance"
                && (streamer.bessPhase != "idle" || streamer.bessJustFinished != nil)
            if inStance {
                BalanceInstructionView(streamer: streamer)
                    .transition(.opacity)
            } else if callStarted, activeTest == nil || activeTest == "balance" {
                // between stances: the three tests and their instructions
                BalanceTestsPanel(streamer: streamer)
                    .padding(.horizontal)
                    .padding(.top, 64)
            }
        }

        // =========================================================
        // STARTUP
        // =========================================================

        .onAppear {

            camera.streamer = streamer
            camera.webRTCClient = webRTC

            camera.shouldStream = false
            camera.streamDepth = false
            camera.shouldSendWebRTC = false
            camera.noseThumbTestActive = false

            configureDoctorCommands()

            camera.start()
        }

        // =========================================================
        // BALANCE / EYE RESULTS
        // =========================================================

        .onChange(
            of: streamer.lastMessage
        ) { _, message in

            guard
                callStarted,
                !message.isEmpty,
                let activeTest,
                let sessionID =
                    activeTestSessionID,
                activeTest == "balance"
                    || activeTest == "eyes"
            else {
                return
            }

            webRTC.signaling
                .sendTestResult(
                    test: activeTest,
                    sessionID: sessionID,
                    rawJSON: message
                )
        }

        // =========================================================
        // NOSE -> THUMB REAL-TIME MEASUREMENT
        //
        // We temporarily keep the backend identifier "finger"
        // until the Python server is updated.
        // =========================================================

        .onChange(
            of: camera.noseThumbMeasurementVersion
        ) { _, _ in

            guard
                callStarted,
                activeTest == "finger",
                let sessionID =
                    activeTestSessionID
            else {
                return
            }

            webRTC.signaling
                .sendFingerMeasurement(
                    sessionID: sessionID,
                    distanceCM:
                        camera.noseThumbDistanceCM
                            .map {
                                Float($0)
                            },
                    handDetected:
                        camera.thumbDetected,
                    landmarks: []
                )
        }

        // =========================================================
        // CLEANUP
        // =========================================================

        .onDisappear {

            stopCurrentTest()

            camera.shouldSendWebRTC = false

            webRTC.disconnect()

            camera.stop()
        }
    }

    // =============================================================
    // DOCTOR COMMANDS
    // =============================================================

    private func configureDoctorCommands() {

        webRTC.signaling.onStartTest = {
            test,
            sessionID in

            DispatchQueue.main.async {

                startTest(
                    test,
                    sessionID: sessionID
                )
            }
        }

        webRTC.signaling.onStopTest = {

            DispatchQueue.main.async {
                stopCurrentTest()
            }
        }

        webRTC.signaling.onServerCommand = {
            test,
            command in

            DispatchQueue.main.async {

                guard activeTest == test else {
                    return
                }

                streamer.sendCommand(
                    command
                )
            }
        }
    }

    // =============================================================
    // CALL
    // =============================================================

    private func startCall() {

        let cleanHost =
            serverIP.trimmingCharacters(
                in: .whitespacesAndNewlines
            )

        let cleanRoom =
            callRoom.trimmingCharacters(
                in: .whitespacesAndNewlines
            )

        guard
            !cleanHost.isEmpty,
            !cleanRoom.isEmpty
        else {
            return
        }

        callStarted = true

        webRTC.connect(
            host: cleanHost,
            room: cleanRoom
        )

        camera.shouldSendWebRTC = true
    }

    private func endCall() {

        stopCurrentTest()

        camera.shouldSendWebRTC = false

        webRTC.disconnect()

        callStarted = false
    }

    // =============================================================
    // START TEST
    // =============================================================

    private func startTest(
        _ test: String,
        sessionID: String
    ) {

        if activeTest != nil {
            stopCurrentTest()
        }

        streamer.disconnect()

        camera.shouldStream = false
        camera.streamDepth = false
        camera.noseThumbTestActive = false

        activeTest = test
        activeTestSessionID = sessionID

        print(
            "Starting test:",
            test,
            "session:",
            sessionID
        )

        // Live only: test videos are no longer recorded or uploaded
        // (the doctor server keeps no history).

        switch test {

        case "finger":

            // This is now the nose-to-thumb NPC measurement.
            // The old backend name is retained temporarily.
            camera.noseThumbTestActive = true

        case "balance":

            camera.streamDepth = true
            camera.shouldStream = true

            streamer.connect(
                host: serverIP,
                port: 8765
            )

        case "eyes":

            camera.shouldStream = true

            streamer.connect(
                host: serverIP,
                port: 8766
            )

        default:

            activeTest = nil
            activeTestSessionID = nil

            return
        }
    }

    // =============================================================
    // STOP TEST
    // =============================================================

    private func stopCurrentTest() {

        guard
            let test = activeTest,
            let sessionID =
                activeTestSessionID
        else {

            camera.shouldStream = false
            camera.streamDepth = false
            camera.noseThumbTestActive = false

            streamer.disconnect()

            return
        }

        print(
            "Stopping test:",
            test,
            "session:",
            sessionID
        )

        camera.shouldStream = false
        camera.streamDepth = false
        camera.noseThumbTestActive = false

        streamer.disconnect()

        activeTest = nil
        activeTestSessionID = nil
    }
}

// =============================================================
// RAW CAMERA PREVIEW
// =============================================================

struct CameraPreview:
    UIViewRepresentable {

    let session:
        AVCaptureSession

    func makeUIView(
        context: Context
    ) -> PreviewContainer {

        let view =
            PreviewContainer()

        view.previewLayer.session =
            session

        view.previewLayer.videoGravity =
            .resizeAspectFill

        return view
    }

    func updateUIView(
        _ uiView: PreviewContainer,
        context: Context
    ) {

        uiView.previewLayer.session =
            session
    }
}

final class PreviewContainer:
    UIView {

    override class var layerClass:
        AnyClass {

        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer:
        AVCaptureVideoPreviewLayer {

        layer as!
            AVCaptureVideoPreviewLayer
    }
}

// =============================================================
// TEST RECORDING MODEL
// =============================================================

struct TestRecording {

    let fileURL: URL
    let test: String
    let sessionID: String
    let durationSeconds: Double
}

enum TestRecordingError:
    LocalizedError {

    case alreadyRecording
    case cannotCreateWriter
    case cannotAddInput
    case recordingFailed
    case noFrames

    var errorDescription:
        String? {

        switch self {

        case .alreadyRecording:

            return
                "A test is already being recorded."

        case .cannotCreateWriter:

            return
                "Could not create the test video writer."

        case .cannotAddInput:

            return
                "Could not add the video recording input."

        case .recordingFailed:

            return
                "The test video recording failed."

        case .noFrames:

            return
                "No camera frames were recorded."
        }
    }
}

// =============================================================
// TEST VIDEO RECORDER
// =============================================================

final class TestVideoRecorder {

    private let queue =
        DispatchQueue(
            label:
                "TeleVision.TestVideoRecorder",
            qos:
                .userInitiated
        )

    private var writer:
        AVAssetWriter?

    private var writerInput:
        AVAssetWriterInput?

    private var pixelBufferAdaptor:
        AVAssetWriterInputPixelBufferAdaptor?

    private var outputURL:
        URL?

    private var currentTest:
        String?

    private var currentSessionID:
        String?

    private var firstTimestamp:
        CMTime?

    private var lastTimestamp:
        CMTime?

    private var frameCount =
        0

    private var recordingRequested =
        false

    private var writerStarted =
        false

    // =========================================================
    // START
    // =========================================================

    func start(
        test: String,
        sessionID: String
    ) {

        queue.sync {

            cleanupWriter()

            let directory =
                FileManager.default
                    .temporaryDirectory
                    .appendingPathComponent(
                        "TeleVisionTests",
                        isDirectory: true
                    )

            try?
                FileManager.default
                    .createDirectory(
                        at: directory,
                        withIntermediateDirectories:
                            true
                    )

            let safeTest =
                test.replacingOccurrences(
                    of: "/",
                    with: "_"
                )

            let filename =
                "\(safeTest)_\(sessionID).mov"

            let url =
                directory
                    .appendingPathComponent(
                        filename
                    )

            try?
                FileManager.default
                    .removeItem(
                        at: url
                    )

            outputURL = url

            currentTest = test
            currentSessionID = sessionID

            firstTimestamp = nil
            lastTimestamp = nil

            frameCount = 0

            recordingRequested = true
            writerStarted = false
        }
    }

    // =========================================================
    // APPEND CAMERA FRAME
    // =========================================================

    func append(
        pixelBuffer: CVPixelBuffer,
        timestamp: CMTime
    ) {

        queue.async {

            guard
                self.recordingRequested
            else {
                return
            }

            if self.writer == nil {

                guard
                    self.configureWriter(
                        using: pixelBuffer
                    )
                else {

                    self.recordingRequested =
                        false

                    return
                }
            }

            guard
                let writer =
                    self.writer,
                let input =
                    self.writerInput,
                let adaptor =
                    self.pixelBufferAdaptor
            else {
                return
            }

            if !self.writerStarted {

                guard
                    writer.startWriting()
                else {

                    print(
                        "AVAssetWriter start error:",
                        writer.error?
                            .localizedDescription
                            ?? "Unknown error"
                    )

                    self.recordingRequested =
                        false

                    return
                }

                writer.startSession(
                    atSourceTime:
                        timestamp
                )

                self.firstTimestamp =
                    timestamp

                self.writerStarted =
                    true
            }

            guard
                input.isReadyForMoreMediaData
            else {
                return
            }

            if adaptor.append(
                pixelBuffer,
                withPresentationTime:
                    timestamp
            ) {

                self.lastTimestamp =
                    timestamp

                self.frameCount += 1

            } else {

                print(
                    "Recording append error:",
                    writer.error?
                        .localizedDescription
                        ?? "Unknown error"
                )
            }
        }
    }

    // =========================================================
    // STOP
    // =========================================================

    func stop(
        completion:
            @escaping (
                Result<
                    TestRecording,
                    Error
                >
            ) -> Void
    ) {

        queue.async {

            guard
                self.recordingRequested
                    || self.writerStarted
            else {

                DispatchQueue.main.async {

                    completion(
                        .failure(
                            TestRecordingError
                                .noFrames
                        )
                    )
                }

                return
            }

            self.recordingRequested =
                false

            guard
                let writer =
                    self.writer,
                let input =
                    self.writerInput,
                let url =
                    self.outputURL,
                let test =
                    self.currentTest,
                let sessionID =
                    self.currentSessionID,
                self.writerStarted,
                self.frameCount > 0
            else {

                self.cleanupWriter()

                DispatchQueue.main.async {

                    completion(
                        .failure(
                            TestRecordingError
                                .noFrames
                        )
                    )
                }

                return
            }

            let first =
                self.firstTimestamp

            let last =
                self.lastTimestamp

            input.markAsFinished()

            writer.finishWriting {

                let status =
                    writer.status

                let writerError =
                    writer.error

                let duration:
                    Double

                if
                    let first,
                    let last
                {

                    let seconds =
                        CMTimeGetSeconds(
                            CMTimeSubtract(
                                last,
                                first
                            )
                        )

                    if seconds.isFinite {

                        duration =
                            max(
                                0,
                                seconds
                            )

                    } else {

                        duration =
                            0
                    }

                } else {

                    duration =
                        0
                }

                self.queue.async {
                    self.cleanupWriter()
                }

                DispatchQueue.main.async {

                    if status == .completed {

                        completion(
                            .success(
                                TestRecording(
                                    fileURL:
                                        url,
                                    test:
                                        test,
                                    sessionID:
                                        sessionID,
                                    durationSeconds:
                                        duration
                                )
                            )
                        )

                    } else {

                        completion(
                            .failure(
                                writerError
                                    ??
                                    TestRecordingError
                                        .recordingFailed
                            )
                        )
                    }
                }
            }
        }
    }

    // =========================================================
    // CANCEL
    // =========================================================

    func cancel() {

        queue.async {

            self.recordingRequested =
                false

            self.writerInput?
                .markAsFinished()

            self.writer?
                .cancelWriting()

            if let url =
                self.outputURL
            {

                try?
                    FileManager.default
                        .removeItem(
                            at: url
                        )
            }

            self.cleanupWriter()
        }
    }

    // =========================================================
    // CONFIGURE WRITER
    // =========================================================

    private func configureWriter(
        using pixelBuffer:
            CVPixelBuffer
    ) -> Bool {

        guard
            let outputURL
        else {
            return false
        }

        let width =
            CVPixelBufferGetWidth(
                pixelBuffer
            )

        let height =
            CVPixelBufferGetHeight(
                pixelBuffer
            )

        guard
            width > 0,
            height > 0
        else {
            return false
        }

        do {

            let writer =
                try AVAssetWriter(
                    outputURL:
                        outputURL,
                    fileType:
                        .mov
                )

            let settings:
                [String: Any] = [

                    AVVideoCodecKey:
                        AVVideoCodecType.hevc,

                    AVVideoWidthKey:
                        width,

                    AVVideoHeightKey:
                        height,

                    AVVideoCompressionPropertiesKey: [

                        AVVideoAverageBitRateKey:
                            8_000_000,

                        AVVideoExpectedSourceFrameRateKey:
                            30,

                        AVVideoMaxKeyFrameIntervalKey:
                            30
                    ]
                ]

            let input =
                AVAssetWriterInput(
                    mediaType:
                        .video,
                    outputSettings:
                        settings
                )

            input
                .expectsMediaDataInRealTime =
                true

            input.transform =
                CGAffineTransform(
                    rotationAngle:
                        .pi / 2
                )
                .translatedBy(
                    x: 0,
                    y: -CGFloat(height)
                )

            guard
                writer.canAdd(
                    input
                )
            else {
                return false
            }

            writer.add(
                input
            )

            let adaptor =
                AVAssetWriterInputPixelBufferAdaptor(
                    assetWriterInput:
                        input,
                    sourcePixelBufferAttributes:
                        nil
                )

            self.writer =
                writer

            self.writerInput =
                input

            self.pixelBufferAdaptor =
                adaptor

            return true

        } catch {

            print(
                "AVAssetWriter creation error:",
                error.localizedDescription
            )

            return false
        }
    }

    // =========================================================
    // CLEANUP
    // =========================================================

    private func cleanupWriter() {

        writer = nil
        writerInput = nil
        pixelBufferAdaptor = nil

        outputURL = nil

        currentTest = nil
        currentSessionID = nil

        firstTimestamp = nil
        lastTimestamp = nil

        frameCount = 0

        recordingRequested = false
        writerStarted = false
    }
}

// =============================================================
// CAMERA MANAGER
// =============================================================

final class CameraManager:
    NSObject,
    ObservableObject,
    AVCaptureDataOutputSynchronizerDelegate {

    // =========================================================
    // NPC / NOSE-THUMB STATE
    // =========================================================

    @Published
    var noseThumbDistanceCM:
        Double?

    @Published
    var personDetected =
        false

    @Published
    var thumbDetected =
        false

    @Published
    var noseThumbMeasurementVersion =
        0

    @Published
    var status =
        "Starting rear LiDAR..."

    // Only run pose/hand inference while the NPC test is active.
    var noseThumbTestActive =
        false

    // =========================================================
    // OTHER PIPELINES
    // =========================================================

    weak var streamer:
        VideoStreamClient?

    var shouldStream =
        false

    // Balance test only: send each frame with its LiDAR depth map
    // (JSON frame, see VideoStreamClient.sendDepthFrame).
    var streamDepth =
        false

    weak var webRTCClient:
        WebRTCClient?

    var shouldSendWebRTC =
        false

    // =========================================================
    // CAMERA
    // =========================================================

    let session =
        AVCaptureSession()

    private let videoOutput =
        AVCaptureVideoDataOutput()

    private let depthOutput =
        AVCaptureDepthDataOutput()

    private var synchronizer:
        AVCaptureDataOutputSynchronizer?

    private let cameraQueue =
        DispatchQueue(
            label:
                "TeleVision.Camera",
            qos:
                .userInteractive
        )

    private var configured =
        false

    private let mediaPipeOrientation:
        UIImage.Orientation = .right

    // =========================================================
    // NOSE / THUMB DETECTOR
    // =========================================================

    private let poseDetector =
        PoseDetector()

    // =========================================================
    // TEST RECORDER
    // =========================================================

    private let testRecorder =
        TestVideoRecorder()

    private var loggedFrameSizes = false   // [FORMAT] diagnostic, log once

    // =========================================================
    // INIT
    // =========================================================

    override init() {

        super.init()

        poseDetector.onMeasurement = {
            [weak self]
            distanceCM,
            personDetected,
            thumbDetected in

            guard
                let self
            else {
                return
            }

            // PoseDetector publishes its result from the main
            // queue, but keep this safe if that ever changes.
            DispatchQueue.main.async {

                self.noseThumbDistanceCM =
                    distanceCM

                self.personDetected =
                    personDetected

                self.thumbDetected =
                    thumbDetected

                self.noseThumbMeasurementVersion
                    &+= 1

                if let distanceCM {

                    self.status =
                        String(
                            format:
                                "Nose to thumb: %.1f cm",
                            distanceCM
                        )

                } else if !personDetected {

                    self.status =
                        "Waiting for face"

                } else if !thumbDetected {

                    self.status =
                        "Face detected • waiting for thumb"

                } else {

                    self.status =
                        "Face + thumb detected • waiting for LiDAR"
                }
            }
        }
    }

    // =========================================================
    // TEST RECORDING
    // =========================================================

    func startTestRecording(
        test: String,
        sessionID: String
    ) {

        testRecorder.start(
            test: test,
            sessionID: sessionID
        )

        print(
            "Started recording:",
            test,
            sessionID
        )
    }

    func stopTestRecording(
        completion:
            @escaping (
                Result<
                    TestRecording,
                    Error
                >
            ) -> Void
    ) {

        testRecorder.stop(
            completion:
                completion
        )
    }

    func cancelTestRecording() {

        testRecorder.cancel()
    }

    // =========================================================
    // CAMERA START
    // =========================================================

    func start() {

        cameraQueue.async {

            if !self.configured {
                self.configureSession()
            }

            guard
                self.configured
            else {
                return
            }

            if !self.session.isRunning {

                self.session
                    .startRunning()
            }

            DispatchQueue.main.async {

                self.status =
                    "Rear LiDAR active"
            }
        }
    }

    // =========================================================
    // CAMERA STOP
    // =========================================================

    func stop() {

        cameraQueue.async {

            if self.session.isRunning {

                self.session
                    .stopRunning()
            }
        }
    }

    // =========================================================
    // CAMERA CONFIGURATION
    // =========================================================

    private func configureSession() {

        session.beginConfiguration()

        defer {
            session.commitConfiguration()
        }

        session.sessionPreset =
            .inputPriority

        guard
            let camera =
                AVCaptureDevice.default(
                    .builtInLiDARDepthCamera,
                    for: .video,
                    position: .back
                )
        else {

            DispatchQueue.main.async {

                self.status =
                    "Rear LiDAR camera unavailable"
            }

            return
        }

        guard
            let best =
                findBestLiDARFormat(
                    camera
                )
        else {

            DispatchQueue.main.async {

                self.status =
                    "No LiDAR depth format found"
            }

            return
        }

        do {

            try camera
                .lockForConfiguration()

            camera.activeFormat =
                best.video

            camera.activeDepthDataFormat =
                best.depth

            let desiredDuration =
                CMTime(
                    value: 1,
                    timescale: 30
                )

            if
                let range =
                    best.video
                        .videoSupportedFrameRateRanges
                        .first,

                range.minFrameRate <= 30,

                range.maxFrameRate >= 30
            {

                camera.activeVideoMinFrameDuration =
                    desiredDuration

                camera.activeVideoMaxFrameDuration =
                    desiredDuration
            }

            // Give PoseDetector the actual selected
            // camera format's FOV.
            let fov =
                camera.activeFormat
                    .videoFieldOfView

            camera.unlockForConfiguration()

            poseDetector
                .setFieldOfView(
                    fov
                )

            print(
                "LiDAR camera FOV:",
                fov
            )

            // [FORMAT] diagnostic: which video/depth formats were picked,
            // and which other formats offer depth (for the depth-streaming check).
            func formatLabel(_ f: AVCaptureDevice.Format) -> String {
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                let r = Double(d.width) / Double(max(d.height, 1))
                let shape = abs(r - 4.0 / 3.0) < 0.01 ? "4:3"
                    : abs(r - 16.0 / 9.0) < 0.01 ? "16:9" : String(format: "%.3f", r)
                return "\(d.width)x\(d.height) (\(shape))"
            }
            print("[FORMAT] chosen video:", formatLabel(best.video),
                  "| chosen depth:", formatLabel(best.depth),
                  "| FOV:", fov)
            for f in camera.formats where !f.supportedDepthDataFormats.isEmpty {
                let depths = f.supportedDepthDataFormats
                    .filter { CMFormatDescriptionGetMediaSubType($0.formatDescription)
                              == kCVPixelFormatType_DepthFloat32 }
                    .map(formatLabel)
                print("[FORMAT] option video:", formatLabel(f), "-> depth Float32:", depths)
            }

            let input =
                try AVCaptureDeviceInput(
                    device: camera
                )

            guard
                session.canAddInput(
                    input
                )
            else {

                DispatchQueue.main.async {

                    self.status =
                        "Cannot add camera input"
                }

                return
            }

            session.addInput(
                input
            )

        } catch {

            DispatchQueue.main.async {

                self.status =
                    "Camera error: \(error.localizedDescription)"
            }

            return
        }

        // -----------------------------------------------------
        // RGB
        // -----------------------------------------------------

        videoOutput.videoSettings = [

            kCVPixelBufferPixelFormatTypeKey
                as String:
                kCVPixelFormatType_32BGRA
        ]

        videoOutput
            .alwaysDiscardsLateVideoFrames =
            true

        guard
            session.canAddOutput(
                videoOutput
            )
        else {

            DispatchQueue.main.async {

                self.status =
                    "Cannot add video output"
            }

            return
        }

        session.addOutput(
            videoOutput
        )

        // -----------------------------------------------------
        // DEPTH
        // -----------------------------------------------------

        depthOutput.isFilteringEnabled =
            true

        depthOutput
            .alwaysDiscardsLateDepthData =
            true

        guard
            session.canAddOutput(
                depthOutput
            )
        else {

            DispatchQueue.main.async {

                self.status =
                    "Cannot add depth output"
            }

            return
        }

        session.addOutput(
            depthOutput
        )

        // -----------------------------------------------------
        // SYNCHRONIZER
        // -----------------------------------------------------

        let sync =
            AVCaptureDataOutputSynchronizer(
                dataOutputs: [
                    videoOutput,
                    depthOutput
                ]
            )

        sync.setDelegate(
            self,
            queue:
                cameraQueue
        )

        synchronizer =
            sync

        configured =
            true
    }

    // =========================================================
    // FIND BEST LIDAR FORMAT
    // =========================================================

    private func findBestLiDARFormat(
        _ device:
            AVCaptureDevice
    ) -> (
        video:
            AVCaptureDevice.Format,
        depth:
            AVCaptureDevice.Format
    )? {

        var bestVideo:
            AVCaptureDevice.Format?

        var bestDepth:
            AVCaptureDevice.Format?

        var bestPixels:
            Int32 = 0

        for videoFormat
            in device.formats
        {

            let videoDimensions =
                CMVideoFormatDescriptionGetDimensions(
                    videoFormat
                        .formatDescription
                )

            guard
                videoDimensions.width <=
                    1920
            else {
                continue
            }

            for depthFormat
                in videoFormat
                    .supportedDepthDataFormats
            {

                let description =
                    depthFormat
                        .formatDescription

                let subtype =
                    CMFormatDescriptionGetMediaSubType(
                        description
                    )

                guard
                    subtype ==
                        kCVPixelFormatType_DepthFloat32
                else {
                    continue
                }

                let dimensions =
                    CMVideoFormatDescriptionGetDimensions(
                        description
                    )

                let pixels =
                    dimensions.width
                    * dimensions.height

                if pixels > bestPixels {

                    bestPixels =
                        pixels

                    bestVideo =
                        videoFormat

                    bestDepth =
                        depthFormat
                }
            }
        }

        guard
            let bestVideo,
            let bestDepth
        else {
            return nil
        }

        return (
            bestVideo,
            bestDepth
        )
    }

    // =========================================================
    // SYNCHRONIZED RGB + LIDAR FRAME
    // =========================================================

    func dataOutputSynchronizer(
        _ synchronizer:
            AVCaptureDataOutputSynchronizer,

        didOutput
            synchronizedDataCollection:
            AVCaptureSynchronizedDataCollection
    ) {

        guard
            let videoData =
                synchronizedDataCollection
                    .synchronizedData(
                        for:
                            videoOutput
                    )
                    as?
                    AVCaptureSynchronizedSampleBufferData,

            let depthData =
                synchronizedDataCollection
                    .synchronizedData(
                        for:
                            depthOutput
                    )
                    as?
                    AVCaptureSynchronizedDepthData
        else {
            return
        }

        guard
            !videoData
                .sampleBufferWasDropped,

            !depthData
                .depthDataWasDropped
        else {
            return
        }

        let sampleBuffer =
            videoData.sampleBuffer

        // [FORMAT] diagnostic: actual buffer sizes of the first frame.
        if !loggedFrameSizes,
           let px = CMSampleBufferGetImageBuffer(sampleBuffer) {
            loggedFrameSizes = true
            let map = depthData.depthData.depthDataMap
            print("[FORMAT] first frame video buffer:",
                  CVPixelBufferGetWidth(px), "x", CVPixelBufferGetHeight(px),
                  "| depth map:", CVPixelBufferGetWidth(map), "x", CVPixelBufferGetHeight(map),
                  "| calibration ref:",
                  depthData.depthData.cameraCalibrationData?
                      .intrinsicMatrixReferenceDimensions as Any)
        }

        let presentationTime =
            CMSampleBufferGetPresentationTimeStamp(
                sampleBuffer
            )

        // =====================================================
        // ORIGINAL RGB FRAME
        // =====================================================

        if let pixelBuffer =
            CMSampleBufferGetImageBuffer(
                sampleBuffer
            )
        {

            // ---------------------------------------------
            // RECORD TEST
            // ---------------------------------------------

            testRecorder.append(
                pixelBuffer:
                    pixelBuffer,
                timestamp:
                    presentationTime
            )

            // ---------------------------------------------
            // BALANCE / EYE PYTHON STREAM
            // ---------------------------------------------

            if shouldStream {

                if streamDepth {

                    streamer?
                        .sendDepthFrame(
                            pixelBuffer,
                            depthData:
                                depthData.depthData,
                            timestamp:
                                presentationTime
                        )

                } else {

                    streamer?
                        .sendFrame(
                            pixelBuffer
                        )
                }
            }

            // ---------------------------------------------
            // WEBRTC
            // ---------------------------------------------

            if shouldSendWebRTC {

                webRTCClient?
                    .sendVideoFrame(
                        pixelBuffer:
                            pixelBuffer,
                        timestamp:
                            presentationTime
                    )
            }
        }

        // =====================================================
        // NPC / NOSE -> THUMB
        // =====================================================

        guard
            noseThumbTestActive
        else {
            return
        }

        /*
         IMPORTANT:

         We pass the original synchronized AVDepthData into
         PoseDetector.

         PoseDetector owns:
           - PoseLandmarker
           - HandLandmarker
           - nose bridge detection
           - thumb tip #4 detection
           - LiDAR sampling
           - 3D nose-to-thumb calculation

         This replaces the old index-finger-to-camera pipeline.
         */

        poseDetector.detect(
            sampleBuffer:
                sampleBuffer,
            depthData:
                depthData.depthData,
            orientation:
                mediaPipeOrientation
        )
    }
}
