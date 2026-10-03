import Foundation
import AVFoundation
import UIKit
import CoreImage
import Combine
import ImageIO
import os

final class VideoStreamClient: ObservableObject {
    @Published var isConnected = false
    @Published var status = "Disconnected"
    @Published var poseDetected = false
    @Published var inferenceMS: Double?
    @Published var lastMessage = ""
    @Published var errorMessage: String?

    // BESS state from the pose server, for the patient's instruction screen
    // (no scores or errors: just which stance and which phase).
    @Published var bessPhase = "idle"              // idle | countdown | running
    @Published var bessStance: String?             // double | tandem | single
    @Published var bessNondominant = "left"
    @Published var bessWaitingForView = false
    @Published var bessCompleted: Set<String> = []
    @Published var bessJustFinished: String?       // shown as "Done" for a few seconds

    private var task: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private let encodingQueue = DispatchQueue(label: "TeleVision.StreamEncoder", qos: .userInitiated)
    private let ciContext = CIContext()
    private let frameSemaphore = DispatchSemaphore(value: 1)
    private let targetInterval: CFTimeInterval = 0.10
    private var lastFrameTime: CFTimeInterval = 0
    private(set) var currentPort: Int?

    func connect(host: String, port: Int) {
        disconnect()

        var cleanHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "ws://", with: "")
            .replacingOccurrences(of: "wss://", with: "")

        if let slash = cleanHost.firstIndex(of: "/") { cleanHost = String(cleanHost[..<slash]) }
        if let colon = cleanHost.firstIndex(of: ":") { cleanHost = String(cleanHost[..<colon]) }

        guard !cleanHost.isEmpty else {
            DispatchQueue.main.async {
                self.status = "Invalid server IP"
                self.errorMessage = "Enter your Mac's IP address."
            }
            return
        }

        guard let url = URL(string: "ws://\(cleanHost):\(port)") else {
            DispatchQueue.main.async { self.status = "Invalid URL" }
            return
        }

        currentPort = port

        DispatchQueue.main.async {
            self.status = "Connecting..."
            self.errorMessage = nil
            self.poseDetected = false
            self.inferenceMS = nil
        }

        let socket = session.webSocketTask(with: url)
        task = socket
        socket.resume()

        socket.sendPing { [weak self] error in
            guard let self else { return }

            if let error {
                DispatchQueue.main.async {
                    self.isConnected = false
                    self.status = "Connection failed"
                    self.errorMessage = error.localizedDescription
                }
                return
            }

            DispatchQueue.main.async {
                self.isConnected = true
                self.status = "Connected • port \(port)"
            }

            self.receiveLoop()
        }
    }

    func disconnect() {
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        currentPort = nil

        DispatchQueue.main.async {
            self.isConnected = false
            self.status = "Disconnected"
            self.poseDetected = false
            self.inferenceMS = nil
            self.bessPhase = "idle"
            self.bessStance = nil
            self.bessWaitingForView = false
            self.bessCompleted = []
            self.bessJustFinished = nil
        }
    }

    // Sends real control messages understood by server.py / eye_server.py.
    func sendCommand(_ command: [String: Any]) {
        guard let task,
              JSONSerialization.isValidJSONObject(command),
              let data = try? JSONSerialization.data(withJSONObject: command),
              let text = String(data: data, encoding: .utf8)
        else { return }

        task.send(.string(text)) { [weak self] error in
            if let error {
                DispatchQueue.main.async { self?.errorMessage = error.localizedDescription }
            }
        }
    }

    func sendFrame(_ pixelBuffer: CVPixelBuffer) {
        guard isConnected else { return }

        let now = CACurrentMediaTime()
        guard now - lastFrameTime >= targetInterval else { return }
        lastFrameTime = now

        guard frameSemaphore.wait(timeout: .now()) == .success else { return }

        encodingQueue.async { [weak self] in
            guard let self else { return }
            defer { self.frameSemaphore.signal() }

            guard let jpeg = self.makeJPEG(from: pixelBuffer),
                  let socket = self.task
            else { return }

            socket.send(.data(jpeg)) { [weak self] error in
                if let error {
                    DispatchQueue.main.async {
                        self?.errorMessage = error.localizedDescription
                        self?.status = "Stream error"
                    }
                }
            }
        }
    }

    // MARK: - Balance frames with LiDAR depth
    //
    // JSON frame in the posecam pose server's format (see posecam/depth.py):
    //   {"type": "frame", "image": <base64 JPEG, NOT rotated>, "rotate": 90,
    //    "depth": <base64 UInt16 millimetres, little-endian>, "depth_format": "uint16_mm",
    //    "depth_size": [w, h], "intrinsics": [fx, fy, cx, cy] (in depth-map pixels),
    //    "camera": "lidar", "frame_id": n, "timestamp_ms": capture time}
    // The server rotates the image and the depth map together. The video and depth
    // formats are both 4:3 (640x480 / 320x240), so they cover the same view.

    private let depthFrameInterval: CFTimeInterval = 1.0 / 15.0   // 15 fps
    // One depth frame in flight. A lock-protected flag instead of a semaphore: the
    // camera thread never waits, and the lock donates priority (no QoS inversion).
    private let depthInFlight = OSAllocatedUnfairLock(initialState: false)
    private let maxDepthWidth = 320
    private var depthFrameIndex = 0

    func sendDepthFrame(_ pixelBuffer: CVPixelBuffer, depthData: AVDepthData, timestamp: CMTime) {
        guard isConnected else { return }
        let now = CACurrentMediaTime()
        guard now - lastFrameTime >= depthFrameInterval else { return }
        let claimed = depthInFlight.withLock { busy -> Bool in
            if busy { return false }
            busy = true
            return true
        }
        guard claimed else { return }
        lastFrameTime = now
        depthFrameIndex += 1
        let frameID = depthFrameIndex
        let timestampMS = CMTimeGetSeconds(timestamp) * 1000

        encodingQueue.async { [weak self] in
            guard let self else { return }
            guard let socket = self.task,
                  let text = self.makeDepthFrameJSON(pixelBuffer, depthData: depthData,
                                                     frameID: frameID, timestampMS: timestampMS)
            else {
                self.depthInFlight.withLock { $0 = false }
                return
            }
            socket.send(.string(text)) { [weak self] error in
                self?.depthInFlight.withLock { $0 = false }
                if let error {
                    DispatchQueue.main.async {
                        self?.errorMessage = error.localizedDescription
                        self?.status = "Stream error"
                    }
                }
            }
        }
    }

    private func makeDepthFrameJSON(_ pixelBuffer: CVPixelBuffer, depthData: AVDepthData,
                                    frameID: Int, timestampMS: Double) -> String? {
        // JPEG in the sensor's orientation (the server applies "rotate")
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        let targetWidth: CGFloat = 640
        if image.extent.width > targetWidth {
            let scale = targetWidth / image.extent.width
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        guard let cgImage = ciContext.createCGImage(image, from: image.extent),
              let jpeg = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.65)
        else { return nil }

        var message: [String: Any] = [
            "type": "frame",
            "frame_id": frameID,
            "timestamp_ms": timestampMS,
            "rotate": 90,
            "camera": "lidar",
            "image": jpeg.base64EncodedString(),
        ]
        // Without calibration the depth can't be placed: send the frame without
        // depth (the server then measures in 2D) rather than dropping it.
        if let depth = encodeDepth(depthData) {
            message["depth"] = depth.data.base64EncodedString()
            message["depth_format"] = "uint16_mm"
            message["depth_size"] = [depth.width, depth.height]
            message["intrinsics"] = depth.intrinsics
        }
        guard let data = try? JSONSerialization.data(withJSONObject: message) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Depth map -> UInt16 millimetres (downsampled to <= maxDepthWidth), plus the
    /// intrinsics scaled to that map. 0 = no depth.
    private func encodeDepth(_ raw: AVDepthData)
        -> (data: Data, width: Int, height: Int, intrinsics: [Double])? {
        let depth = raw.depthDataType == kCVPixelFormatType_DepthFloat32
            ? raw : raw.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = depth.depthDataMap
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
                let z = row[x * step]                                   // metres
                out[y * ow + x] = (z.isFinite && z > 0) ? UInt16(min(z * 1000, 65535)) : 0
            }
        }
        let bytes = out.withUnsafeBufferPointer { Data(buffer: $0) }   // iOS is little-endian

        guard let cal = depth.cameraCalibrationData else { return nil }
        let m = cal.intrinsicMatrix                                     // column-major 3x3
        let ref = cal.intrinsicMatrixReferenceDimensions
        let sx = Double(w) / Double(ref.width) / Double(step)
        let sy = Double(h) / Double(ref.height) / Double(step)
        let k = [Double(m.columns.0.x) * sx, Double(m.columns.1.y) * sy,
                 Double(m.columns.2.x) * sx, Double(m.columns.2.y) * sy]
        return (bytes, ow, oh, k)
    }

    private func makeJPEG(from pixelBuffer: CVPixelBuffer) -> Data? {
        var image = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        let extent = image.extent
        guard extent.width > 0 else { return nil }

        let targetWidth: CGFloat = 640
        if extent.width > targetWidth {
            let scale = targetWidth / extent.width
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }

        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.65)
    }

    private func applyBess(_ b: [String: Any]) {
        let phase = b["phase"] as? String ?? "idle"
        if bessPhase == "running", phase == "idle", let finished = bessStance {
            bessJustFinished = finished
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) { [weak self] in
                if self?.bessJustFinished == finished { self?.bessJustFinished = nil }
            }
        }
        if phase != "idle" { bessJustFinished = nil }
        bessPhase = phase
        bessStance = b["stance"] as? String
        if let nd = b["nondominant"] as? String { bessNondominant = nd }
        bessWaitingForView = b["waiting_for_view"] as? Bool ?? false
        if let scores = (b["session"] as? [String: Any])?["scores"] as? [String: Any] {
            bessCompleted = Set(scores.compactMap { $0.value is NSNull ? nil : $0.key })
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self else { return }

            switch result {
            case .failure(let error):
                DispatchQueue.main.async {
                    self.isConnected = false
                    self.status = "Disconnected"
                    self.errorMessage = error.localizedDescription
                }

            case .success(let message):
                switch message {
                case .string(let text):
                    self.handleJSON(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) { self.handleJSON(text) }
                @unknown default:
                    break
                }
                self.receiveLoop()
            }
        }
    }

    private func handleJSON(_ text: String) {
        DispatchQueue.main.async { self.lastMessage = text }

        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        let type = json["type"] as? String ?? ""

        DispatchQueue.main.async {
            if type == "pose" {
                if let detected = json["detected"] as? Bool { self.poseDetected = detected }
                if let ms = json["inference_ms"] as? NSNumber { self.inferenceMS = ms.doubleValue }
                if let bess = json["bess"] as? [String: Any] { self.applyBess(bess) }
            } else if type == "error", let message = json["message"] as? String {
                self.errorMessage = message
            }
        }
    }
}
