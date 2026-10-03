import Foundation
import Combine
import CoreGraphics

final class WebRTCSignalingClient: NSObject, ObservableObject {

    @Published var isConnected = false
    @Published var status = "Disconnected"

    @Published var offlineAnalysisStatus = ""
    @Published var uploadProgress: Double = 0

    // =========================================================
    // CALLBACKS
    // =========================================================

    var onOffer: ((String) -> Void)?
    var onAnswer: ((String) -> Void)?

    var onIceCandidate:
        ((String, Int32, String?) -> Void)?

    // IMPORTANT:
    // Server now supplies the session ID.
    //
    // test, sessionID
    var onStartTest:
        ((String, String) -> Void)?

    var onStopTest:
        (() -> Void)?

    var onServerCommand:
        ((String, [String: Any]) -> Void)?

    var onOfflineAnalysisComplete:
        (([String: Any]) -> Void)?

    // =========================================================
    // INTERNAL STATE
    // =========================================================

    private var webSocket:
        URLSessionWebSocketTask?

    private var session:
        URLSession?

    private var host = ""
    private var port = 8088
    private var room = ""
    private var role = ""

    private var manuallyDisconnected = false

    // =========================================================
    // CONNECT
    // =========================================================

    func connect(
        host: String,
        port: Int,
        room: String,
        role: String
    ) {

        disconnect()

        self.host = cleanHost(host)
        self.port = port
        self.room = room
        self.role = role

        manuallyDisconnected = false

        guard
            !self.host.isEmpty,
            let url = URL(
                string:
                    "ws://\(self.host):\(port)/ws"
            )
        else {

            DispatchQueue.main.async {

                self.status =
                    "Invalid signaling URL"

                self.isConnected =
                    false
            }

            return
        }

        let configuration =
            URLSessionConfiguration.default

        configuration.timeoutIntervalForRequest =
            15

        let session =
            URLSession(
                configuration:
                    configuration
            )

        self.session =
            session

        let socket =
            session.webSocketTask(
                with: url
            )

        self.webSocket =
            socket

        DispatchQueue.main.async {

            self.status =
                "Connecting..."

            self.isConnected =
                false
        }

        socket.resume()

        receiveLoop()

        // Joining immediately after resume is okay:
        // URLSession queues the WebSocket send while
        // the connection is being established.
        send([
            "type": "join",
            "room": room,
            "role": role
        ])
    }

    // =========================================================
    // DISCONNECT
    // =========================================================

    func disconnect() {

        manuallyDisconnected =
            true

        webSocket?
            .cancel(
                with: .normalClosure,
                reason: nil
            )

        webSocket =
            nil

        session?
            .invalidateAndCancel()

        session =
            nil

        DispatchQueue.main.async {

            self.isConnected =
                false

            self.status =
                "Disconnected"
        }
    }

    // =========================================================
    // RECEIVE LOOP
    // =========================================================

    private func receiveLoop() {

        guard
            let webSocket
        else {
            return
        }

        webSocket.receive {
            [weak self]
            result in

            guard let self else {
                return
            }

            switch result {

            case .success(let message):

                self.handleMessage(
                    message
                )

                if !self.manuallyDisconnected {

                    self.receiveLoop()
                }

            case .failure(let error):

                guard
                    !self.manuallyDisconnected
                else {
                    return
                }

                DispatchQueue.main.async {

                    self.isConnected =
                        false

                    self.status =
                        "Signaling error: \(error.localizedDescription)"
                }
            }
        }
    }

    // =========================================================
    // HANDLE SERVER MESSAGE
    // =========================================================

    private func handleMessage(
        _ message:
            URLSessionWebSocketTask.Message
    ) {

        let data: Data

        switch message {

        case .string(let string):

            guard
                let value =
                    string.data(
                        using: .utf8
                    )
            else {
                return
            }

            data = value

        case .data(let value):

            data = value

        @unknown default:
            return
        }

        guard
            let object =
                try? JSONSerialization
                    .jsonObject(
                        with: data
                    ),

            let json =
                object
                    as? [String: Any],

            let type =
                json["type"]
                    as? String
        else {
            return
        }

        switch type {

        // -----------------------------------------------------
        // JOINED
        // -----------------------------------------------------

        case "joined":

            DispatchQueue.main.async {

                self.isConnected =
                    true

                self.status =
                    "Signaling connected"
            }

        // -----------------------------------------------------
        // PEER JOINED
        // -----------------------------------------------------

        case "peer_joined":

            DispatchQueue.main.async {

                self.status =
                    "Doctor connected"
            }

        // -----------------------------------------------------
        // PEER LEFT
        // -----------------------------------------------------

        case "peer_left":

            DispatchQueue.main.async {

                self.status =
                    "Doctor disconnected"
            }

        // -----------------------------------------------------
        // WEBRTC OFFER
        // -----------------------------------------------------

        case "offer":

            guard
                let sdp =
                    json["sdp"]
                        as? String
            else {
                return
            }

            DispatchQueue.main.async {

                self.onOffer?(
                    sdp
                )
            }

        // -----------------------------------------------------
        // WEBRTC ANSWER
        // -----------------------------------------------------

        case "answer":

            guard
                let sdp =
                    json["sdp"]
                        as? String
            else {
                return
            }

            DispatchQueue.main.async {

                self.onAnswer?(
                    sdp
                )
            }

        // -----------------------------------------------------
        // ICE
        // -----------------------------------------------------

        case "ice":

            guard
                let candidate =
                    json["candidate"]
                        as? String
            else {
                return
            }

            let index: Int32

            if let number =
                json["sdpMLineIndex"]
                    as? NSNumber {

                index =
                    number.int32Value

            } else if let number =
                json["sdpMLineIndex"]
                    as? Int {

                index =
                    Int32(number)

            } else {

                index = 0
            }

            let mid =
                json["sdpMid"]
                    as? String

            DispatchQueue.main.async {

                self.onIceCandidate?(
                    candidate,
                    index,
                    mid
                )
            }

        // -----------------------------------------------------
        // START TEST
        //
        // NEW PROTOCOL:
        //
        // {
        //   "type": "start_test",
        //   "test": "finger",
        //   "session_id": "..."
        // }
        // -----------------------------------------------------

        case "start_test":

            guard
                let test =
                    json["test"]
                        as? String,

                let sessionID =
                    json["session_id"]
                        as? String,

                !sessionID.isEmpty
            else {

                print(
                    "start_test missing test or session_id:",
                    json
                )

                return
            }

            DispatchQueue.main.async {

                self.onStartTest?(
                    test,
                    sessionID
                )
            }

        // -----------------------------------------------------
        // STOP TEST
        // -----------------------------------------------------

        case "stop_test":

            DispatchQueue.main.async {

                self.onStopTest?()
            }

        // -----------------------------------------------------
        // PYTHON ANALYZER COMMAND
        // -----------------------------------------------------

        case "server_command":

            guard
                let test =
                    json["test"]
                        as? String,

                let command =
                    json["command"]
                        as? [String: Any]
            else {
                return
            }

            DispatchQueue.main.async {

                self.onServerCommand?(
                    test,
                    command
                )
            }

        // -----------------------------------------------------
        // TEST SAVED
        // -----------------------------------------------------

        case "test_saved":

            DispatchQueue.main.async {

                self.offlineAnalysisStatus =
                    "Waiting for recording..."
            }

        // -----------------------------------------------------
        // OFFLINE ANALYSIS QUEUED
        // -----------------------------------------------------

        case "offline_analysis_queued":

            DispatchQueue.main.async {

                self.offlineAnalysisStatus =
                    "Offline analysis processing..."
            }

        // -----------------------------------------------------
        // OFFLINE ANALYSIS COMPLETE
        // -----------------------------------------------------

        case "offline_analysis_complete":

            DispatchQueue.main.async {

                self.offlineAnalysisStatus =
                    "Offline analysis complete"

                self.uploadProgress =
                    1.0

                self.onOfflineAnalysisComplete?(
                    json
                )
            }

        // -----------------------------------------------------
        // OFFLINE ANALYSIS FAILED
        // -----------------------------------------------------

        case "offline_analysis_failed":

            let serverMessage =
                json["message"]
                    as? String
                ?? "Unknown error"

            DispatchQueue.main.async {

                self.offlineAnalysisStatus =
                    "Offline analysis failed: \(serverMessage)"
            }

        default:
            break
        }
    }

    // =========================================================
    // WEBRTC ANSWER
    // =========================================================

    func sendAnswer(
        _ sdp: String
    ) {

        send([
            "type": "answer",
            "room": room,
            "role": role,
            "sdp": sdp
        ])
    }

    // =========================================================
    // WEBRTC OFFER
    // =========================================================

    func sendOffer(
        _ sdp: String
    ) {

        send([
            "type": "offer",
            "room": room,
            "role": role,
            "sdp": sdp
        ])
    }

    // =========================================================
    // ICE
    // =========================================================

    func sendIceCandidate(
        sdp: String,
        sdpMLineIndex: Int32,
        sdpMid: String?
    ) {

        var message:
            [String: Any] = [

                "type": "ice",
                "room": room,
                "role": role,

                "candidate":
                    sdp,

                "sdpMLineIndex":
                    Int(
                        sdpMLineIndex
                    )
            ]

        if let sdpMid {

            message["sdpMid"] =
                sdpMid
        }

        send(
            message
        )
    }

    // =========================================================
    // REAL-TIME FINGER RESULT
    // =========================================================

    func sendFingerMeasurement(
        sessionID: String,
        distanceCM: Float?,
        handDetected: Bool,
        landmarks: [CGPoint]
    ) {

        var payload:
            [String: Any] = [

                "type":
                    "measurement",

                "test":
                    "finger",

                "session_id":
                    sessionID,

                "hand_detected":
                    handDetected,

                "landmarks":
                    landmarks.map {

                        [
                            "x":
                                Double(
                                    $0.x
                                ),

                            "y":
                                Double(
                                    $0.y
                                )
                        ]
                    },

                "timestamp_ms":
                    Int64(
                        Date()
                            .timeIntervalSince1970
                            * 1000
                    )
            ]

        if let distanceCM {

            payload[
                "distance_cm"
            ] =
                Double(
                    distanceCM
                )

        } else {

            payload[
                "distance_cm"
            ] =
                NSNull()
        }

        send(
            payload
        )
    }

    // =========================================================
    // REAL-TIME BALANCE / EYES RESULT
    // =========================================================

    func sendTestResult(
        test: String,
        sessionID: String,
        rawJSON: String
    ) {

        guard
            let data =
                rawJSON.data(
                    using: .utf8
                ),

            let result =
                try? JSONSerialization
                    .jsonObject(
                        with: data
                    )
        else {
            return
        }

        send([
            "type":
                "measurement",

            "test":
                test,

            "session_id":
                sessionID,

            "result":
                result,

            "timestamp_ms":
                Int64(
                    Date()
                        .timeIntervalSince1970
                        * 1000
                )
        ])
    }

    // =========================================================
    // TEST VIDEO UPLOAD
    //
    // Matches doctor_call_server.py:
    //
    // POST /api/upload-test
    //
    // multipart/form-data
    //   room
    //   test
    //   session_id
    //   video
    // =========================================================

    func uploadTestRecording(
        fileURL: URL,
        room: String,
        test: String,
        sessionID: String,
        host: String
    ) {

        let clean =
            cleanHost(
                host
            )

        guard
            !clean.isEmpty,
            let url =
                URL(
                    string:
                        "http://\(clean):8088/api/upload-test"
                )
        else {

            DispatchQueue.main.async {

                self.offlineAnalysisStatus =
                    "Invalid upload URL"
            }

            return
        }

        DispatchQueue.main.async {

            self.offlineAnalysisStatus =
                "Uploading test recording..."

            self.uploadProgress =
                0
        }

        let boundary =
            "TeleVision-\(UUID().uuidString)"

        var request =
            URLRequest(
                url: url
            )

        request.httpMethod =
            "POST"

        request.timeoutInterval =
            300

        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField:
                "Content-Type"
        )

        DispatchQueue.global(
            qos: .userInitiated
        ).async {

            do {

                // -------------------------------------------------
                // TEMP MULTIPART BODY FILE
                //
                // We do NOT load the entire MOV into RAM.
                // -------------------------------------------------

                let bodyURL =
                    FileManager.default
                        .temporaryDirectory
                        .appendingPathComponent(
                            "television-upload-\(UUID().uuidString).tmp"
                        )

                FileManager.default
                    .createFile(
                        atPath:
                            bodyURL.path,
                        contents:
                            nil
                    )

                let output =
                    try FileHandle(
                        forWritingTo:
                            bodyURL
                    )

                // -------------------------------------------------
                // ROOM
                // -------------------------------------------------

                try self.writeMultipartField(
                    name:
                        "room",

                    value:
                        room,

                    boundary:
                        boundary,

                    to:
                        output
                )

                // -------------------------------------------------
                // TEST
                // -------------------------------------------------

                try self.writeMultipartField(
                    name:
                        "test",

                    value:
                        test,

                    boundary:
                        boundary,

                    to:
                        output
                )

                // -------------------------------------------------
                // SESSION ID
                // -------------------------------------------------

                try self.writeMultipartField(
                    name:
                        "session_id",

                    value:
                        sessionID,

                    boundary:
                        boundary,

                    to:
                        output
                )

                // -------------------------------------------------
                // VIDEO PART HEADER
                // -------------------------------------------------

                let filename =
                    fileURL.lastPathComponent

                let videoHeader =
                    "--\(boundary)\r\n"
                    +
                    "Content-Disposition: form-data; "
                    +
                    "name=\"video\"; "
                    +
                    "filename=\"\(filename)\"\r\n"
                    +
                    "Content-Type: video/quicktime\r\n"
                    +
                    "\r\n"

                if let headerData =
                    videoHeader.data(
                        using: .utf8
                    ) {

                    try output.write(
                        contentsOf:
                            headerData
                    )
                }

                // -------------------------------------------------
                // COPY MOV INTO MULTIPART FILE
                // -------------------------------------------------

                let input =
                    try FileHandle(
                        forReadingFrom:
                            fileURL
                    )

                while true {

                    let chunk =
                        try input.read(
                            upToCount:
                                1_048_576
                        )

                    guard
                        let chunk,
                        !chunk.isEmpty
                    else {
                        break
                    }

                    try output.write(
                        contentsOf:
                            chunk
                    )
                }

                try input.close()

                // -------------------------------------------------
                // MULTIPART FOOTER
                // -------------------------------------------------

                let footer =
                    "\r\n--\(boundary)--\r\n"

                if let footerData =
                    footer.data(
                        using: .utf8
                    ) {

                    try output.write(
                        contentsOf:
                            footerData
                    )
                }

                try output.synchronize()

                try output.close()

                // -------------------------------------------------
                // UPLOAD
                // -------------------------------------------------

                let configuration =
                    URLSessionConfiguration.default

                configuration.timeoutIntervalForRequest =
                    300

                configuration.timeoutIntervalForResource =
                    600

                let uploadSession =
                    URLSession(
                        configuration:
                            configuration
                    )

                let uploadTask =
                    uploadSession.uploadTask(
                        with:
                            request,

                        fromFile:
                            bodyURL
                    ) {
                        data,
                        response,
                        error in

                        defer {

                            try?
                            FileManager.default
                                .removeItem(
                                    at:
                                        bodyURL
                                )

                            uploadSession
                                .finishTasksAndInvalidate()
                        }

                        // -----------------------------------------
                        // NETWORK ERROR
                        // -----------------------------------------

                        if let error {

                            DispatchQueue.main.async {

                                self.offlineAnalysisStatus =
                                    "Upload failed: \(error.localizedDescription)"
                            }

                            return
                        }

                        guard
                            let http =
                                response
                                    as?
                                    HTTPURLResponse
                        else {

                            DispatchQueue.main.async {

                                self.offlineAnalysisStatus =
                                    "Invalid upload response"
                            }

                            return
                        }

                        // -----------------------------------------
                        // SERVER ERROR
                        // -----------------------------------------

                        guard
                            200..<300
                                ~= http.statusCode
                        else {

                            var serverMessage =
                                "HTTP \(http.statusCode)"

                            if
                                let data,
                                let text =
                                    String(
                                        data:
                                            data,

                                        encoding:
                                            .utf8
                                    ),

                                !text.isEmpty
                            {

                                serverMessage =
                                    text
                            }

                            DispatchQueue.main.async {

                                self.offlineAnalysisStatus =
                                    "Upload failed: \(serverMessage)"
                            }

                            return
                        }

                        // -----------------------------------------
                        // SUCCESS
                        // -----------------------------------------

                        DispatchQueue.main.async {

                            self.uploadProgress =
                                1.0

                            self.offlineAnalysisStatus =
                                "Offline analysis processing..."
                        }

                        // Server now owns the recording.
                        // Delete the local temporary MOV.

                        try?
                        FileManager.default
                            .removeItem(
                                at:
                                    fileURL
                            )
                    }

                uploadTask.resume()

            } catch {

                DispatchQueue.main.async {

                    self.offlineAnalysisStatus =
                        "Upload failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // =========================================================
    // MULTIPART TEXT FIELD
    // =========================================================

    private func writeMultipartField(
        name: String,
        value: String,
        boundary: String,
        to handle: FileHandle
    ) throws {

        let string =
            "--\(boundary)\r\n"
            +
            "Content-Disposition: form-data; "
            +
            "name=\"\(name)\"\r\n"
            +
            "\r\n"
            +
            "\(value)\r\n"

        guard
            let data =
                string.data(
                    using: .utf8
                )
        else {
            return
        }

        try handle.write(
            contentsOf:
                data
        )
    }

    // =========================================================
    // GENERIC WEBSOCKET SEND
    // =========================================================

    private func send(
        _ dictionary:
            [String: Any]
    ) {

        guard
            JSONSerialization
                .isValidJSONObject(
                    dictionary
                )
        else {

            print(
                "Invalid signaling JSON:",
                dictionary
            )

            return
        }

        do {

            let data =
                try JSONSerialization
                    .data(
                        withJSONObject:
                            dictionary
                    )

            guard
                let string =
                    String(
                        data:
                            data,

                        encoding:
                            .utf8
                    )
            else {
                return
            }

            guard
                let webSocket
            else {

                print(
                    "WebSocket send skipped: socket is nil"
                )

                return
            }

            webSocket.send(
                .string(
                    string
                )
            ) {
                error in

                if let error {

                    print(
                        "WebSocket send error:",
                        error.localizedDescription
                    )
                }
            }

        } catch {

            print(
                "Signaling serialization error:",
                error.localizedDescription
            )
        }
    }

    // =========================================================
    // CLEAN HOST
    // =========================================================

    private func cleanHost(
        _ value: String
    ) -> String {

        var result =
            value
                .trimmingCharacters(
                    in:
                        .whitespacesAndNewlines
                )
                .replacingOccurrences(
                    of:
                        "http://",
                    with:
                        ""
                )
                .replacingOccurrences(
                    of:
                        "https://",
                    with:
                        ""
                )
                .replacingOccurrences(
                    of:
                        "ws://",
                    with:
                        ""
                )
                .replacingOccurrences(
                    of:
                        "wss://",
                    with:
                        ""
                )

        if let slash =
            result.firstIndex(
                of: "/"
            ) {

            result =
                String(
                    result[
                        ..<slash
                    ]
                )
        }

        if let colon =
            result.firstIndex(
                of: ":"
            ) {

            result =
                String(
                    result[
                        ..<colon
                    ]
                )
        }

        return result
    }
}
