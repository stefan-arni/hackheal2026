import Foundation
import WebRTC
import Combine
import AVFoundation
import CoreMedia
import CoreVideo

final class WebRTCClient: NSObject, ObservableObject {

    @Published var connectionStatus = "Not connected"
    @Published var isConnected = false
    @Published var remoteVideoTrack: RTCVideoTrack?

    let signaling = WebRTCSignalingClient()

    private let factory: RTCPeerConnectionFactory

    private var peerConnection: RTCPeerConnection?

    private var localAudioTrack: RTCAudioTrack?
    private var localAudioSource: RTCAudioSource?

    private var localVideoSource: RTCVideoSource?
    private var localVideoTrack: RTCVideoTrack?
    private var externalVideoCapturer: RTCVideoCapturer?

    // =========================================================
    // INIT
    // =========================================================

    override init() {

        RTCInitializeSSL()

        let encoderFactory =
            RTCDefaultVideoEncoderFactory()

        let decoderFactory =
            RTCDefaultVideoDecoderFactory()

        self.factory =
            RTCPeerConnectionFactory(
                encoderFactory:
                    encoderFactory,

                decoderFactory:
                    decoderFactory
            )

        super.init()

        configureSignalingCallbacks()

        configureAudioSession()
    }

    // =========================================================
    // DEINIT
    // =========================================================

    deinit {

        peerConnection?.close()

        RTCCleanupSSL()
    }

    // =========================================================
    // CONNECT
    // =========================================================

    func connect(
        host: String,
        room: String
    ) {

        disconnect()

        createPeerConnection()

        signaling.connect(
            host: host,
            port: 8088,
            room: room,
            role: "patient"
        )

        DispatchQueue.main.async {

            self.connectionStatus =
                "Connecting..."
        }
    }

    // =========================================================
    // DISCONNECT
    // =========================================================

    func disconnect() {

        signaling.disconnect()

        localAudioTrack?
            .isEnabled =
            false

        localVideoTrack?
            .isEnabled =
            false

        peerConnection?
            .close()

        peerConnection =
            nil

        localAudioTrack =
            nil

        localAudioSource =
            nil

        localVideoTrack =
            nil

        localVideoSource =
            nil

        externalVideoCapturer =
            nil

        DispatchQueue.main.async {

            self.remoteVideoTrack =
                nil

            self.isConnected =
                false

            self.connectionStatus =
                "Disconnected"
        }
    }

    /// Close the current peer connection and build a new one (same steps as
    /// connect(), without touching the signaling connection).
    private func resetPeerConnection() {

        peerConnection?.close()
        peerConnection = nil

        localAudioTrack = nil
        localAudioSource = nil
        localVideoTrack = nil
        localVideoSource = nil
        externalVideoCapturer = nil

        DispatchQueue.main.async {
            self.remoteVideoTrack = nil
        }

        createPeerConnection()
    }

    // =========================================================
    // AUDIO SESSION
    // =========================================================

    private func configureAudioSession() {

        let audioSession =
            RTCAudioSession
                .sharedInstance()

        audioSession
            .lockForConfiguration()

        do {

            try audioSession.setCategory(
                .playAndRecord,
                with: [
                    .allowBluetooth,
                    .defaultToSpeaker
                ]
            )

            try audioSession.setMode(
                .videoChat
            )

        } catch {

            print(
                "WebRTC audio configuration error:",
                error.localizedDescription
            )
        }

        audioSession
            .unlockForConfiguration()
    }

    // =========================================================
    // SIGNALING CALLBACKS
    // =========================================================

    private func configureSignalingCallbacks() {

        signaling.onOffer = {
            [weak self]
            sdp in

            self?
                .handleRemoteOffer(
                    sdp
                )
        }

        signaling.onAnswer = {
            [weak self]
            sdp in

            self?
                .handleRemoteAnswer(
                    sdp
                )
        }

        signaling.onIceCandidate = {
            [weak self]
            candidate,
            index,
            mid in

            self?
                .handleRemoteIceCandidate(
                    candidate,
                    sdpMLineIndex:
                        index,
                    sdpMid:
                        mid
                )
        }
    }

    // =========================================================
    // CREATE PEER CONNECTION
    // =========================================================

    private func createPeerConnection() {

        let configuration =
            RTCConfiguration()

        configuration.iceServers = [

            RTCIceServer(
                urlStrings: [
                    "stun:stun.l.google.com:19302"
                ]
            )
        ]

        configuration.sdpSemantics =
            .unifiedPlan

        let constraints =
            RTCMediaConstraints(
                mandatoryConstraints:
                    nil,

                optionalConstraints:
                    nil
            )

        guard
            let connection =
                factory.peerConnection(
                    with:
                        configuration,

                    constraints:
                        constraints,

                    delegate:
                        self
                )
        else {

            DispatchQueue.main.async {

                self.connectionStatus =
                    "Could not create WebRTC connection"
            }

            return
        }

        peerConnection =
            connection

        createLocalAudioTrack()

        createLocalVideoTrack()
    }

    // =========================================================
    // AUDIO TRACK
    // =========================================================

    private func createLocalAudioTrack() {

        let constraints =
            RTCMediaConstraints(
                mandatoryConstraints:
                    nil,

                optionalConstraints: [

                    "googEchoCancellation":
                        "true",

                    "googNoiseSuppression":
                        "true",

                    "googAutoGainControl":
                        "true"
                ]
            )

        let source =
            factory.audioSource(
                with:
                    constraints
            )

        localAudioSource =
            source

        let track =
            factory.audioTrack(
                with:
                    source,

                trackId:
                    "TeleVisionAudio"
            )

        track.isEnabled =
            true

        localAudioTrack =
            track

        peerConnection?
            .add(
                track,
                streamIds: [
                    "TeleVisionStream"
                ]
            )
    }

    // =========================================================
    // VIDEO TRACK
    // =========================================================

    private func createLocalVideoTrack() {

        let source =
            factory.videoSource()

        localVideoSource =
            source

        externalVideoCapturer =
            RTCVideoCapturer(
                delegate:
                    source
            )

        let track =
            factory.videoTrack(
                with:
                    source,

                trackId:
                    "TeleVisionVideo"
            )

        track.isEnabled =
            true

        localVideoTrack =
            track

        peerConnection?
            .add(
                track,
                streamIds: [
                    "TeleVisionStream"
                ]
            )
    }

    // =========================================================
    // CAMERA → WEBRTC
    //
    // IMPORTANT:
    //
    // This receives the SAME original BGRA CVPixelBuffer
    // produced by CameraManager.
    //
    // There is:
    //
    // - no MediaPipe drawing
    // - no overlay renderer
    // - no CIImage render pass
    // - no pixel-buffer copy
    //
    // The RTC rotation metadata handles orientation.
    // =========================================================

    func sendVideoFrame(
        pixelBuffer:
            CVPixelBuffer,

        timestamp:
            CMTime
    ) {

        guard
            let localVideoSource,
            let externalVideoCapturer
        else {
            return
        }

        let seconds =
            CMTimeGetSeconds(
                timestamp
            )

        guard
            seconds.isFinite
        else {
            return
        }

        let rtcPixelBuffer =
            RTCCVPixelBuffer(
                pixelBuffer:
                    pixelBuffer
            )

        let timestampNS =
            Int64(
                seconds
                * 1_000_000_000
            )

        let frame =
            RTCVideoFrame(
                buffer:
                    rtcPixelBuffer,

                rotation:
                    ._90,

                timeStampNs:
                    timestampNS
            )

        localVideoSource
            .capturer(
                externalVideoCapturer,
                didCapture:
                    frame
            )
    }

    // =========================================================
    // REMOTE OFFER
    // =========================================================

    private func handleRemoteOffer(
        _ sdp: String
    ) {

        // A new offer after the call was already set up means the doctor's
        // page reloaded or reconnected with a fresh connection: start a fresh
        // one here too (applying it to the old connection fails).
        if peerConnection?.remoteDescription != nil {
            resetPeerConnection()
        }

        guard
            let peerConnection
        else {
            return
        }

        let description =
            RTCSessionDescription(
                type:
                    .offer,

                sdp:
                    sdp
            )

        peerConnection
            .setRemoteDescription(
                description
            ) {
                [weak self]
                error in

                if let error {

                    DispatchQueue.main.async {

                        self?
                            .connectionStatus =
                            "Offer error: \(error.localizedDescription)"
                    }

                    return
                }

                self?
                    .createAnswer()
            }
    }

    // =========================================================
    // CREATE ANSWER
    // =========================================================

    private func createAnswer() {

        guard
            let peerConnection
        else {
            return
        }

        let constraints =
            RTCMediaConstraints(
                mandatoryConstraints: [

                    "OfferToReceiveAudio":
                        "true",

                    "OfferToReceiveVideo":
                        "true"
                ],

                optionalConstraints:
                    nil
            )

        peerConnection.answer(
            for:
                constraints
        ) {
            [weak self]
            description,
            error in

            guard
                let self
            else {
                return
            }

            if let error {

                DispatchQueue.main.async {

                    self.connectionStatus =
                        "Answer error: \(error.localizedDescription)"
                }

                return
            }

            guard
                let description
            else {
                return
            }

            peerConnection
                .setLocalDescription(
                    description
                ) {
                    [weak self]
                    error in

                    guard
                        let self
                    else {
                        return
                    }

                    if let error {

                        DispatchQueue.main.async {

                            self.connectionStatus =
                                "Local SDP error: \(error.localizedDescription)"
                        }

                        return
                    }

                    self.signaling
                        .sendAnswer(
                            description.sdp
                        )
                }
        }
    }

    // =========================================================
    // REMOTE ANSWER
    // =========================================================

    private func handleRemoteAnswer(
        _ sdp: String
    ) {

        guard
            let peerConnection
        else {
            return
        }

        let description =
            RTCSessionDescription(
                type:
                    .answer,

                sdp:
                    sdp
            )

        peerConnection
            .setRemoteDescription(
                description
            ) {
                [weak self]
                error in

                if let error {

                    DispatchQueue.main.async {

                        self?
                            .connectionStatus =
                            "Remote answer error: \(error.localizedDescription)"
                    }
                }
            }
    }

    // =========================================================
    // REMOTE ICE
    // =========================================================

    private func handleRemoteIceCandidate(
        _ candidate:
            String,

        sdpMLineIndex:
            Int32,

        sdpMid:
            String?
    ) {

        guard
            let peerConnection
        else {
            return
        }

        let iceCandidate =
            RTCIceCandidate(
                sdp:
                    candidate,

                sdpMLineIndex:
                    sdpMLineIndex,

                sdpMid:
                    sdpMid
            )

        peerConnection.add(
            iceCandidate
        )
    }
}


// =============================================================
// RTCPeerConnectionDelegate
// =============================================================

extension WebRTCClient:
    RTCPeerConnectionDelegate {

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didChange stateChanged:
            RTCSignalingState
    ) {
    }

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didAdd stream:
            RTCMediaStream
    ) {

        if let track =
            stream.videoTracks.first {

            DispatchQueue.main.async {

                self.remoteVideoTrack =
                    track
            }
        }
    }

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didRemove stream:
            RTCMediaStream
    ) {

        DispatchQueue.main.async {

            self.remoteVideoTrack =
                nil
        }
    }

    func peerConnectionShouldNegotiate(
        _ peerConnection:
            RTCPeerConnection
    ) {
    }

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didChange newState:
            RTCIceConnectionState
    ) {

        DispatchQueue.main.async {

            switch newState {

            case .new:

                self.connectionStatus =
                    "Starting..."

            case .checking:

                self.connectionStatus =
                    "Connecting..."

            case .connected,
                 .completed:

                self.connectionStatus =
                    "Connected"

                self.isConnected =
                    true

            case .disconnected:

                self.connectionStatus =
                    "Connection interrupted"

                self.isConnected =
                    false

            case .failed:

                self.connectionStatus =
                    "Connection failed"

                self.isConnected =
                    false

            case .closed:

                self.connectionStatus =
                    "Disconnected"

                self.isConnected =
                    false

            case .count:

                break

            @unknown default:

                break
            }
        }
    }

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didChange newState:
            RTCIceGatheringState
    ) {
    }

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didGenerate candidate:
            RTCIceCandidate
    ) {

        signaling
            .sendIceCandidate(
                sdp:
                    candidate.sdp,

                sdpMLineIndex:
                    candidate
                        .sdpMLineIndex,

                sdpMid:
                    candidate
                        .sdpMid
            )
    }

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didRemove candidates:
            [RTCIceCandidate]
    ) {
    }

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didOpen dataChannel:
            RTCDataChannel
    ) {
    }

    // =========================================================
    // UNIFIED PLAN REMOTE TRACK
    // =========================================================

    func peerConnection(
        _ peerConnection:
            RTCPeerConnection,

        didStartReceivingOn transceiver:
            RTCRtpTransceiver
    ) {

        if let videoTrack =
            transceiver
                .receiver
                .track
                as? RTCVideoTrack {

            DispatchQueue.main.async {

                self.remoteVideoTrack =
                    videoTrack
            }
        }
    }
}
