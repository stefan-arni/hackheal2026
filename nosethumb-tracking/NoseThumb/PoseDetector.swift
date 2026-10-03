//
//  PoseDetector.swift
//  NoseThumb
//

import AVFoundation
import UIKit
import Observation
import MediaPipeTasksVision

// One body point from MediaPipe, copied into our own simple type.
nonisolated struct TrackedPoint: Sendable {
    let x: CGFloat          // 0...1 across the image width
    let y: CGFloat          // 0...1 down the image height
    let visibility: Float   // how sure MediaPipe is that the point is visible
}

// Runs MediaPipe Pose Landmarker (nose, shoulders) and Hand Landmarker (thumb)
// on live camera frames, and reads their depth from the TrueDepth depth map.
// Frames go in through `detect(sampleBuffer:depthData:)`; results come back
// asynchronously in the delegate method at the bottom.
@Observable
nonisolated final class PoseDetector: NSObject, PoseLandmarkerLiveStreamDelegate, HandLandmarkerLiveStreamDelegate, @unchecked Sendable {
    // Model files in the app bundle (without ".task"). Pose options: lite, full, heavy.
    static let modelName = "pose_landmarker_full"
    static let handModelName = "hand_landmarker"

    // Hand landmark index from the MediaPipe docs (0 = wrist, 1-4 = thumb base to tip).
    static let thumbTipIndex = 4

    // Points below this visibility count as "not visible".
    static let minVisibility: Float = 0.5

    // If a frame has no depth, reuse the last valid depth if it's at most this old.
    // The head barely moves during the test, so a second is safe.
    static let depthHoldMs = 1000

    // Pose landmark indices from the MediaPipe docs.
    static let noseIndex = 0
    static let leftEyeInnerIndex = 1
    static let rightEyeInnerIndex = 4
    static let leftShoulderIndex = 11
    static let rightShoulderIndex = 12

    // Shown on screen. Only changed on the main thread.
    private(set) var isLoaded = false
    private(set) var handLoaded = false
    private(set) var personDetected = false
    private(set) var nose: TrackedPoint?
    private(set) var bridge: TrackedPoint?       // midpoint of the inner eye corners (inside the face)
    private(set) var bridgeFront: TrackedPoint?  // front edge of the nose bridge, found in the depth map
    private(set) var leftShoulder: TrackedPoint?
    private(set) var rightShoulder: TrackedPoint?
    private(set) var imageSize: CGSize = .zero   // camera frame size in pixels
    private(set) var bridgeDepthCM: Double?      // distance phone -> bridge front, nil if no valid depth
    private(set) var bridgeStatus = ""           // temporary debug: how the bridge was found, or why not
    private(set) var isProfile = false           // face is side-on to the camera (needed for the test)
    private(set) var thumb: TrackedPoint?        // thumb tip, nil if no hand found
    private(set) var thumbDepthCM: Double?       // distance phone -> thumb tip, nil if no valid depth
    private(set) var fieldOfView: Float = 0      // camera field of view (degrees, along the image's long side)

    // Bridge of the nose -> thumb tip distance in cm, or nil if anything needed is missing.
    // Side view: the bridge and the thumb are both on the midline, roughly the same distance
    // from the phone. So we measure in that plane: pixel distance x bridge depth / focal length.
    // (Thumb-tip depth is not used: a thin, moving fingertip often reads the wall behind it.)
    var bridgeThumbCM: Double? {
        guard let bridgeFront, let bridgeZ = bridgeDepthCM,
              let thumb, fieldOfView > 0, imageSize != .zero else { return nil }

        let width = Double(imageSize.width)
        let height = Double(imageSize.height)
        // Focal length in pixels: half the long side divided by tan(half the field of view).
        let focal = (max(width, height) / 2) / tan(Double(fieldOfView) * .pi / 180 / 2)

        // Distance between the two dots in image pixels.
        let dx = (Double(bridgeFront.x) - Double(thumb.x)) * width
        let dy = (Double(bridgeFront.y) - Double(thumb.y)) * height
        let pixelDistance = (dx * dx + dy * dy).squareRoot()

        // At depth Z, one pixel covers Z / focal cm.
        return pixelDistance * bridgeZ / focal
    }
    private(set) var depthAccuracy = ""          // "absolute" or "relative", as reported by the camera

    @ObservationIgnored private var landmarker: PoseLandmarker?
    @ObservationIgnored private var handLandmarker: HandLandmarker?

    // Last valid depth for each point, to cover frames without depth.
    // Each is only touched from its own MediaPipe result callback.
    @ObservationIgnored private var lastBridgeDepth: (cm: Double, timestampMs: Int)?
    // How far in front of the eye-corner midpoint the bridge edge was last found (0...1 image
    // units), so a frame where the scan fails can reuse it and the dot still follows the head.
    @ObservationIgnored private var lastBridgeOffset: (dx: CGFloat, timestampMs: Int)?
    @ObservationIgnored private var lastThumbDepth: (cm: Double, timestampMs: Int)?
    @ObservationIgnored private var lastImageSize: CGSize = .zero   // used on the video queue

    // Depth maps waiting for their MediaPipe result, keyed by frame timestamp (ms).
    // Written on the video queue, read on MediaPipe's thread, so guarded by a lock.
    @ObservationIgnored private var pendingDepth: [Int: AVDepthData] = [:]
    @ObservationIgnored private let pendingDepthLock = NSLock()

    override init() {
        super.init()
        setUpPose()
        setUpHand()
    }

    private func setUpPose() {
        guard let modelPath = Bundle.main.path(forResource: Self.modelName, ofType: "task") else {
            print("Pose model file \(Self.modelName).task not found in app bundle")
            return
        }

        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        options.numPoses = 1
        options.poseLandmarkerLiveStreamDelegate = self

        do {
            landmarker = try PoseLandmarker(options: options)
            isLoaded = true
        } catch {
            print("Could not create PoseLandmarker: \(error)")
        }
    }

    private func setUpHand() {
        guard let modelPath = Bundle.main.path(forResource: Self.handModelName, ofType: "task") else {
            print("Hand model file \(Self.handModelName).task not found in app bundle")
            return
        }

        let options = HandLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        options.numHands = 1
        options.handLandmarkerLiveStreamDelegate = self

        do {
            handLandmarker = try HandLandmarker(options: options)
            handLoaded = true
        } catch {
            print("Could not create HandLandmarker: \(error)")
        }
    }

    // Called on the camera's video queue for every frame.
    func detect(sampleBuffer: CMSampleBuffer, depthData: AVDepthData?) {
        guard landmarker != nil || handLandmarker != nil else { return }

        // Remember the frame size so the overlay can place dots correctly.
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                              height: CVPixelBufferGetHeight(pixelBuffer))
            if size != lastImageSize {
                lastImageSize = size
                DispatchQueue.main.async {
                    self.imageSize = size
                }
            }
        }

        // Frames are already rotated upright by the camera connection, so "up" is correct.
        guard let image = try? MPImage(sampleBuffer: sampleBuffer, orientation: .up) else { return }

        // MediaPipe needs a timestamp that always increases. Use the frame's own capture time.
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestampMs = Int(CMTimeGetSeconds(time) * 1000)

        // Keep this frame's depth map until both models return their results.
        // Both results need it, so instead of removing on use, drop maps older than 1 second.
        pendingDepthLock.withLock {
            pendingDepth = pendingDepth.filter { $0.key > timestampMs - 1000 }
            if let depthData {
                pendingDepth[timestampMs] = depthData
            }
        }

        // The same frame goes to both models.
        do {
            try landmarker?.detectAsync(image: image, timestampInMilliseconds: timestampMs)
        } catch {
            print("Pose detectAsync failed: \(error)")
        }
        do {
            try handLandmarker?.detectAsync(image: image, timestampInMilliseconds: timestampMs)
        } catch {
            print("Hand detectAsync failed: \(error)")
        }
    }

    // This frame's depth map. Some frames arrive without one (depth can run slower than video),
    // so otherwise use the newest earlier map, if it's at most `maxDepthAgeMs` old. The head
    // barely moves in that time.
    static let maxDepthAgeMs = 200

    private func depthData(for timestampMs: Int) -> AVDepthData? {
        pendingDepthLock.withLock {
            if let exact = pendingDepth[timestampMs] {
                return exact
            }
            let newestEarlier = pendingDepth.keys
                .filter { $0 < timestampMs && $0 >= timestampMs - Self.maxDepthAgeMs }
                .max()
            return newestEarlier.flatMap { pendingDepth[$0] }
        }
    }

    // Called by MediaPipe (on its own background thread) when a frame is done.
    func poseLandmarker(_ poseLandmarker: PoseLandmarker,
                        didFinishDetection result: PoseLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        if let error {
            print("Pose detection error: \(error)")
        }

        // First (and only) person, if any.
        let person = result?.landmarks.first
        let nose = person.flatMap { Self.point($0, Self.noseIndex) }
        let bridge = person.flatMap { Self.bridgePoint($0) }
        let leftShoulder = person.flatMap { Self.point($0, Self.leftShoulderIndex) }
        let rightShoulder = person.flatMap { Self.point($0, Self.rightShoulderIndex) }
        let found = person != nil

        let depthData = depthData(for: timestampInMilliseconds)

        // Find the bridge of the nose and its depth.
        var bridgeFront: TrackedPoint?
        var measuredBridge: Double?
        var bridgeStatus: String
        var profile = false
        if let person, let nose, nose.visibility >= Self.minVisibility,
           let bridge, bridge.visibility >= Self.minVisibility {
            profile = Self.isProfile(person, nose: nose, bridge: bridge)
            if !profile {
                // Facing the camera: the eye-corner midpoint is already on the bridge's skin.
                bridgeFront = bridge
                lastBridgeOffset = nil
                if let depthData, let meters = Self.depthMeters(at: bridge, in: depthData) {
                    measuredBridge = Double(meters) * 100
                    bridgeStatus = "bridge: frontal ✓"
                } else {
                    bridgeStatus = "bridge: frontal, no depth"
                }
            } else if let depthData {
                // Side view: the midpoint is inside the face; scan forward to the front edge.
                // The nose tip is only used to know which way is "forward".
                switch Self.findBridgeFront(from: bridge, toward: nose, in: depthData) {
                case .found(let point, let meters):
                    bridgeFront = point
                    measuredBridge = Double(meters) * 100
                    lastBridgeOffset = (point.x - bridge.x, timestampInMilliseconds)
                    bridgeStatus = "bridge: profile ✓"
                case .failed(let reason):
                    // Reuse the last offset for a short while so the dot doesn't flicker.
                    // The depth for this frame comes from holdDepth below.
                    if let last = lastBridgeOffset, timestampInMilliseconds - last.timestampMs <= Self.depthHoldMs {
                        bridgeFront = TrackedPoint(x: bridge.x + last.dx, y: bridge.y, visibility: bridge.visibility)
                        bridgeStatus = "bridge: profile, held (\(reason))"
                    } else {
                        bridgeStatus = "bridge: profile, \(reason)"
                    }
                }
            } else {
                bridgeStatus = "bridge: profile, no depth this frame"
            }
        } else {
            bridgeStatus = "bridge: face not visible"
        }
        let bridgeDepthCM = bridgeFront == nil ? nil
            : Self.holdDepth(measuredBridge, last: &lastBridgeDepth, timestampMs: timestampInMilliseconds)
        let accuracy = depthData.map { $0.depthDataAccuracy == .absolute ? "absolute" : "relative" }

        DispatchQueue.main.async {
            self.personDetected = found
            self.nose = nose
            self.bridge = bridge
            self.bridgeFront = bridgeFront
            self.bridgeStatus = bridgeStatus
            self.isProfile = profile
            self.leftShoulder = leftShoulder
            self.rightShoulder = rightShoulder
            self.bridgeDepthCM = bridgeDepthCM
            if let accuracy { self.depthAccuracy = accuracy }
        }
    }

    // Called by MediaPipe (on its own background thread) when the hand model is done with a frame.
    func handLandmarker(_ handLandmarker: HandLandmarker,
                        didFinishDetection result: HandLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        if let error {
            print("Hand detection error: \(error)")
        }

        // The hand model has no per-point visibility; a returned hand already passed its
        // confidence check, so its points count as visible.
        let hand = result?.landmarks.first
        let thumb = hand.flatMap { Self.point($0, Self.thumbTipIndex, defaultVisibility: 1) }

        // A fingertip is small: the patch often includes background behind it. The thumb is
        // the closest thing in the patch, so use the near end (25th percentile), not the middle.
        var measuredThumb: Double?
        if let thumb, let depthData = depthData(for: timestampInMilliseconds),
           let meters = Self.depthMeters(at: thumb, in: depthData, radius: 2, percentile: 0.25) {
            measuredThumb = Double(meters) * 100
        }
        let thumbDepthCM = thumb == nil ? nil
            : Self.holdDepth(measuredThumb, last: &lastThumbDepth, timestampMs: timestampInMilliseconds)

        DispatchQueue.main.async {
            self.thumb = thumb
            self.thumbDepthCM = thumbDepthCM
        }
    }

    // Midpoint of the two inner eye corners = bridge of the nose.
    // From the side the far eye is hidden and only estimated, so the point counts as visible
    // if at least one inner eye corner is visible.
    private static func bridgePoint(_ landmarks: [NormalizedLandmark]) -> TrackedPoint? {
        guard let left = point(landmarks, leftEyeInnerIndex),
              let right = point(landmarks, rightEyeInnerIndex) else { return nil }
        return TrackedPoint(x: (left.x + right.x) / 2,
                            y: (left.y + right.y) / 2,
                            visibility: max(left.visibility, right.visibility))
    }

    // Called once by the camera after it picks its format.
    func setFieldOfView(_ degrees: Float) {
        DispatchQueue.main.async {
            self.fieldOfView = degrees
        }
    }

    // Returns the new depth if valid (and remembers it); otherwise the last valid depth
    // if it's recent enough; otherwise nil.
    private static func holdDepth(_ measured: Double?, last: inout (cm: Double, timestampMs: Int)?,
                                  timestampMs: Int) -> Double? {
        if let measured {
            last = (measured, timestampMs)
            return measured
        }
        if let last, timestampMs - last.timestampMs <= depthHoldMs {
            return last.cm
        }
        return nil
    }

    // Depth (meters) in a small patch around a point: the given percentile of the valid
    // pixels (0.5 = median). Skips invalid pixels (NaN, zero, infinite).
    // Returns nil if the patch has no valid depth.
    private static func depthMeters(at point: TrackedPoint, in depthData: AVDepthData,
                                    radius: Int = 3, percentile: Double = 0.5) -> Float? {
        withDepthMap(depthData) { map in
            // The depth map covers the same view as the video, so 0...1 coordinates map directly.
            let centerX = Int(point.x * CGFloat(map.width))
            let centerY = Int(point.y * CGFloat(map.height))
            guard (0..<map.width).contains(centerX), (0..<map.height).contains(centerY) else { return nil }

            var values: [Float] = []
            for y in max(0, centerY - radius)...min(map.height - 1, centerY + radius) {
                for x in max(0, centerX - radius)...min(map.width - 1, centerX + radius) {
                    if let value = map.value(x, y) {
                        values.append(value)
                    }
                }
            }
            guard !values.isEmpty else { return nil }
            values.sort()
            let index = min(values.count - 1, Int(Double(values.count) * percentile))
            return values[index]
        }
    }

    // Walks along the bridge's row in the depth map, toward the front of the face (the side the
    // nose tip is on), until the depth jumps from "face" to "background". Returns the last face
    // pixel before that jump (the front edge of the nose bridge) and the face depth there (meters).
    // Returns nil if no clear edge is found within a few cm.
    // Result of the bridge scan: the point and its depth (meters), or why it failed.
    private enum BridgeScan {
        case found(point: TrackedPoint, meters: Float)
        case failed(String)
    }

    // Facing the camera, the two inner eye corners are far apart sideways and the nose tip sits
    // between them. From the side, the eye corners nearly overlap and the nose tip sticks out
    // to one side. So: profile if the nose tip is farther from the midpoint than the eyes are apart.
    private static func isProfile(_ landmarks: [NormalizedLandmark], nose: TrackedPoint, bridge: TrackedPoint) -> Bool {
        guard let left = point(landmarks, leftEyeInnerIndex),
              let right = point(landmarks, rightEyeInnerIndex) else { return false }
        let eyeSeparation = abs(left.x - right.x)
        let noseOffset = abs(nose.x - bridge.x)
        return noseOffset > eyeSeparation
    }

    private static func findBridgeFront(from bridge: TrackedPoint, toward nose: TrackedPoint,
                                        in depthData: AVDepthData) -> BridgeScan {
        // Starting face depth: closest valid value in a 7x7 patch around the start, so one
        // missing pixel (eyes and lashes often have none) doesn't stop the scan.
        guard let startDepth = depthMeters(at: bridge, in: depthData, radius: 3, percentile: 0) else {
            return .failed("no start depth")
        }

        let result: BridgeScan? = withDepthMap(depthData) { map in
            let startX = Int(bridge.x * CGFloat(map.width))
            let y = Int(bridge.y * CGFloat(map.height))
            guard (0..<map.width).contains(startX), (1..<(map.height - 1)).contains(y) else {
                return .failed("start outside image")
            }
            let step = nose.x >= bridge.x ? 1 : -1

            // Depth of one column: the closest valid value in a 3-pixel-tall band (less noise).
            func columnDepth(_ x: Int) -> Float? {
                [map.value(x, y - 1), map.value(x, y), map.value(x, y + 1)].compactMap { $0 }.min()
            }

            let backgroundJump: Float = 0.10   // 10 cm farther than the face = background
            // About 9 cm at 50 cm from the phone. Depth maps differ in size between cameras,
            // so scale with the map width (80 px on the front camera's 480-wide map).
            let maxSteps = max(20, map.width * 80 / 480)
            let edgeRun = 3                    // background must last 3 pixels in a row to count

            var faceDepths: [Float] = [startDepth]
            var lastFaceX = startX
            var backgroundCount = 0
            var x = startX
            for _ in 0..<maxSteps {
                x += step
                guard (0..<map.width).contains(x) else { return .failed("reached image edge") }
                if let depth = columnDepth(x), depth < startDepth + backgroundJump {
                    lastFaceX = x
                    faceDepths.append(depth)
                    backgroundCount = 0
                } else {
                    backgroundCount += 1
                    if backgroundCount >= edgeRun {
                        // Face depth at the edge: median of the last few face pixels.
                        let recent = faceDepths.suffix(5).sorted()
                        let meters = recent[recent.count / 2]
                        let point = TrackedPoint(x: (CGFloat(lastFaceX) + 0.5) / CGFloat(map.width),
                                                 y: bridge.y,
                                                 visibility: bridge.visibility)
                        return .found(point: point, meters: meters)
                    }
                }
            }
            return .failed("no edge within \(maxSteps) px")
        }
        return result ?? .failed("depth map unreadable")
    }

    // Share (0...1) of depth-map pixels that hold a valid distance. Temporary debug.
    static func validDepthFraction(_ depthData: AVDepthData) -> Double? {
        withDepthMap(depthData) { map in
            var valid = 0
            for y in 0..<map.height {
                for x in 0..<map.width where map.value(x, y) != nil {
                    valid += 1
                }
            }
            return Double(valid) / Double(map.width * map.height)
        }
    }

    // Read-only view of a depth map in meters. `value` returns nil for invalid pixels
    // (NaN, zero, infinite) and for coordinates outside the map.
    private struct DepthMap {
        let base: UnsafeMutableRawPointer
        let width: Int
        let height: Int
        let bytesPerRow: Int

        func value(_ x: Int, _ y: Int) -> Float? {
            guard (0..<width).contains(x), (0..<height).contains(y) else { return nil }
            let value = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)[x]
            return value.isFinite && value > 0 ? value : nil
        }
    }

    // Converts the depth data to Float32 meters, locks it for reading, and runs `body` on it.
    private static func withDepthMap<T>(_ depthData: AVDepthData, _ body: (DepthMap) -> T?) -> T? {
        let depth = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData
            : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let pixelBuffer = depth.depthDataMap

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        return body(DepthMap(base: base,
                             width: CVPixelBufferGetWidth(pixelBuffer),
                             height: CVPixelBufferGetHeight(pixelBuffer),
                             bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer)))
    }

    // Copies one MediaPipe landmark into a TrackedPoint.
    // `defaultVisibility` is used when the model gives no visibility score.
    private static func point(_ landmarks: [NormalizedLandmark], _ index: Int,
                              defaultVisibility: Float = 0) -> TrackedPoint? {
        guard index < landmarks.count else { return nil }
        let landmark = landmarks[index]
        return TrackedPoint(x: CGFloat(landmark.x),
                            y: CGFloat(landmark.y),
                            visibility: landmark.visibility?.floatValue ?? defaultVisibility)
    }
}
