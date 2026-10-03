import AVFoundation
import UIKit
import Observation
import MediaPipeTasksVision

// =============================================================
// TRACKED POINT
// =============================================================

nonisolated struct TrackedPoint: Sendable {
    let x: CGFloat
    let y: CGFloat
    let visibility: Float
}

// =============================================================
// NOSE / THUMB DETECTOR
// =============================================================

@Observable
nonisolated final class PoseDetector:
    NSObject,
    PoseLandmarkerLiveStreamDelegate,
    HandLandmarkerLiveStreamDelegate,
    @unchecked Sendable {

    // =========================================================
    // MODELS
    // =========================================================

    static let modelName = "pose_landmarker_full"
    static let handModelName = "hand_landmarker"

    // =========================================================
    // LANDMARK INDICES
    // =========================================================

    static let thumbTipIndex = 4

    static let noseIndex = 0
    static let leftEyeInnerIndex = 1
    static let rightEyeInnerIndex = 4

    static let leftShoulderIndex = 11
    static let rightShoulderIndex = 12

    // =========================================================
    // DEPTH TIMING
    // =========================================================

    static let depthHoldMs = 1000
    static let maxDepthAgeMs = 200

    // =========================================================
    // PUBLIC RESULTS
    // =========================================================

    private(set) var isLoaded = false
    private(set) var handLoaded = false

    private(set) var personDetected = false

    private(set) var nose: TrackedPoint?
    private(set) var bridge: TrackedPoint?
    private(set) var bridgeFront: TrackedPoint?

    private(set) var leftShoulder: TrackedPoint?
    private(set) var rightShoulder: TrackedPoint?

    private(set) var thumb: TrackedPoint?

    private(set) var imageSize: CGSize = .zero

    private(set) var bridgeDepthCM: Double?
    private(set) var thumbDepthCM: Double?

    private(set) var bridgeStatus = ""

    private(set) var isProfile = false

    private(set) var fieldOfView: Float = 0

    private(set) var depthAccuracy = ""

    // =========================================================
    // RESULT CALLBACK
    // =========================================================

    // distanceCM, personDetected, thumbDetected
    var onMeasurement:
        ((Double?, Bool, Bool) -> Void)?

    // =========================================================
    // TRUE 3D NOSE-BRIDGE -> THUMB DISTANCE
    // =========================================================

    var bridgeThumbCM: Double? {

        guard
            let bridgePoint = bridgeFront ?? bridge,
            let thumb,
            let bridgeDepthCM,
            let thumbDepthCM,
            fieldOfView > 0,
            imageSize.width > 0,
            imageSize.height > 0
        else {
            return nil
        }

        let width =
            Double(imageSize.width)

        let height =
            Double(imageSize.height)

        let fovRadians =
            Double(fieldOfView)
            * .pi
            / 180.0

        let focalPixels =
            max(width, height)
            /
            (
                2.0
                * tan(
                    fovRadians / 2.0
                )
            )

        guard
            focalPixels.isFinite,
            focalPixels > 0
        else {
            return nil
        }

        let cx =
            width / 2.0

        let cy =
            height / 2.0

        // -----------------------------------------------------
        // NOSE BRIDGE 3D POSITION
        // -----------------------------------------------------

        let bridgeU =
            Double(bridgePoint.x)
            * width

        let bridgeV =
            Double(bridgePoint.y)
            * height

        let bridgeZ =
            bridgeDepthCM

        let bridgeX =
            (bridgeU - cx)
            * bridgeZ
            / focalPixels

        let bridgeY =
            (bridgeV - cy)
            * bridgeZ
            / focalPixels

        // -----------------------------------------------------
        // THUMB 3D POSITION
        // -----------------------------------------------------

        let thumbU =
            Double(thumb.x)
            * width

        let thumbV =
            Double(thumb.y)
            * height

        let thumbZ =
            thumbDepthCM

        let thumbX =
            (thumbU - cx)
            * thumbZ
            / focalPixels

        let thumbY =
            (thumbV - cy)
            * thumbZ
            / focalPixels

        // -----------------------------------------------------
        // EUCLIDEAN 3D DISTANCE
        // -----------------------------------------------------

        let dx =
            thumbX - bridgeX

        let dy =
            thumbY - bridgeY

        let dz =
            thumbZ - bridgeZ

        let distance =
            sqrt(
                dx * dx
                + dy * dy
                + dz * dz
            )

        guard
            distance.isFinite,
            distance > 0
        else {
            return nil
        }

        return distance
    }

    // =========================================================
    // MEDIAPIPE
    // =========================================================

    @ObservationIgnored
    private var landmarker:
        PoseLandmarker?

    @ObservationIgnored
    private var handLandmarker:
        HandLandmarker?

    // =========================================================
    // LAST VALID DEPTH
    // =========================================================

    @ObservationIgnored
    private var lastBridgeDepth:
        (
            cm: Double,
            timestampMs: Int
        )?

    @ObservationIgnored
    private var lastBridgeOffset:
        (
            dx: CGFloat,
            timestampMs: Int
        )?

    @ObservationIgnored
    private var lastThumbDepth:
        (
            cm: Double,
            timestampMs: Int
        )?

    @ObservationIgnored
    private var lastImageSize:
        CGSize = .zero

    // =========================================================
    // PENDING DEPTH FRAMES
    // =========================================================

    @ObservationIgnored
    private var pendingDepth:
        [Int: AVDepthData] = [:]

    @ObservationIgnored
    private let pendingDepthLock =
        NSLock()

    // =========================================================
    // INIT
    // =========================================================

    override init() {

        super.init()

        setUpPose()
        setUpHand()
    }

    // =========================================================
    // SETUP POSE
    // =========================================================

    private func setUpPose() {

        guard
            let modelPath =
                Bundle.main.path(
                    forResource:
                        Self.modelName,
                    ofType:
                        "task"
                )
        else {

            print(
                "Pose model \(Self.modelName).task not found"
            )

            return
        }

        let options =
            PoseLandmarkerOptions()

        options
            .baseOptions
            .modelAssetPath =
            modelPath

        options.runningMode =
            .liveStream

        options.numPoses =
            1

        options
            .poseLandmarkerLiveStreamDelegate =
            self

        do {

            landmarker =
                try PoseLandmarker(
                    options:
                        options
                )

            isLoaded =
                true

            print(
                "Pose Landmarker loaded"
            )

        } catch {

            print(
                "Could not create PoseLandmarker:",
                error.localizedDescription
            )
        }
    }

    // =========================================================
    // SETUP HAND
    // =========================================================

    private func setUpHand() {

        guard
            let modelPath =
                Bundle.main.path(
                    forResource:
                        Self.handModelName,
                    ofType:
                        "task"
                )
        else {

            print(
                "Hand model \(Self.handModelName).task not found"
            )

            return
        }

        let options =
            HandLandmarkerOptions()

        options
            .baseOptions
            .modelAssetPath =
            modelPath

        options.runningMode =
            .liveStream

        options.numHands =
            1

        options
            .minHandDetectionConfidence =
            0.5

        options
            .minHandPresenceConfidence =
            0.5

        options
            .minTrackingConfidence =
            0.5

        options
            .handLandmarkerLiveStreamDelegate =
            self

        do {

            handLandmarker =
                try HandLandmarker(
                    options:
                        options
                )

            handLoaded =
                true

            print(
                "Hand Landmarker loaded"
            )

        } catch {

            print(
                "Could not create HandLandmarker:",
                error.localizedDescription
            )
        }
    }

    // =========================================================
    // CAMERA FIELD OF VIEW
    // =========================================================

    func setFieldOfView(
        _ degrees: Float
    ) {

        DispatchQueue.main.async {

            self.fieldOfView =
                degrees
        }
    }

    // =========================================================
    // PROCESS SYNCHRONIZED FRAME
    // =========================================================

    func detect(
        sampleBuffer:
            CMSampleBuffer,

        depthData:
            AVDepthData?,

        orientation:
            UIImage.Orientation = .right
    ) {

        guard
            landmarker != nil
                || handLandmarker != nil
        else {
            return
        }

        // -----------------------------------------------------
        // IMAGE SIZE
        // -----------------------------------------------------

        if let pixelBuffer =
            CMSampleBufferGetImageBuffer(
                sampleBuffer
            )
        {

            let rawWidth =
                CVPixelBufferGetWidth(
                    pixelBuffer
                )

            let rawHeight =
                CVPixelBufferGetHeight(
                    pixelBuffer
                )

            let size:
                CGSize

            if
                orientation == .right
                    || orientation == .left
            {

                size =
                    CGSize(
                        width:
                            rawHeight,
                        height:
                            rawWidth
                    )

            } else {

                size =
                    CGSize(
                        width:
                            rawWidth,
                        height:
                            rawHeight
                    )
            }

            if size !=
                lastImageSize
            {

                lastImageSize =
                    size

                DispatchQueue.main.async {

                    self.imageSize =
                        size
                }
            }
        }

        // -----------------------------------------------------
        // MEDIAPIPE IMAGE
        // -----------------------------------------------------

        guard
            let image =
                try? MPImage(
                    sampleBuffer:
                        sampleBuffer,
                    orientation:
                        orientation
                )
        else {
            return
        }

        // -----------------------------------------------------
        // TIMESTAMP
        // -----------------------------------------------------

        let time =
            CMSampleBufferGetPresentationTimeStamp(
                sampleBuffer
            )

        var timestampMs =
            Int(
                CMTimeGetSeconds(
                    time
                )
                * 1000
            )

        if timestampMs < 0 {
            timestampMs = 0
        }

        // -----------------------------------------------------
        // STORE DEPTH
        // -----------------------------------------------------

        pendingDepthLock.lock()

        pendingDepth =
            pendingDepth.filter {

                $0.key >
                    timestampMs - 1000
            }

        if let depthData {

            pendingDepth[
                timestampMs
            ] =
                depthData
        }

        pendingDepthLock.unlock()

        // -----------------------------------------------------
        // POSE
        // -----------------------------------------------------

        do {

            try landmarker?
                .detectAsync(
                    image:
                        image,
                    timestampInMilliseconds:
                        timestampMs
                )

        } catch {

            print(
                "Pose detectAsync failed:",
                error.localizedDescription
            )
        }

        // -----------------------------------------------------
        // HAND
        // -----------------------------------------------------

        do {

            try handLandmarker?
                .detectAsync(
                    image:
                        image,
                    timestampInMilliseconds:
                        timestampMs
                )

        } catch {

            print(
                "Hand detectAsync failed:",
                error.localizedDescription
            )
        }
    }

    // =========================================================
    // DEPTH FRAME LOOKUP
    // =========================================================

    private func depthData(
        for timestampMs:
            Int
    ) -> AVDepthData? {

        pendingDepthLock.lock()

        defer {
            pendingDepthLock.unlock()
        }

        if let exact =
            pendingDepth[
                timestampMs
            ]
        {
            return exact
        }

        let newestEarlier =
            pendingDepth
                .keys
                .filter {

                    $0 < timestampMs
                    &&
                    $0 >=
                        timestampMs
                        - Self.maxDepthAgeMs
                }
                .max()

        if let newestEarlier {

            return pendingDepth[
                newestEarlier
            ]
        }

        return nil
    }

    // =========================================================
    // POSE RESULT
    // =========================================================

    func poseLandmarker(
        _ poseLandmarker:
            PoseLandmarker,

        didFinishDetection
            result:
            PoseLandmarkerResult?,

        timestampInMilliseconds:
            Int,

        error:
            Error?
    ) {

        if let error {

            print(
                "Pose detection error:",
                error.localizedDescription
            )
        }

        guard
            let person =
                result?
                    .landmarks
                    .first
        else {

            DispatchQueue.main.async {

                self.personDetected =
                    false

                self.nose =
                    nil

                self.bridge =
                    nil

                self.bridgeFront =
                    nil

                self.leftShoulder =
                    nil

                self.rightShoulder =
                    nil

                self.bridgeDepthCM =
                    nil

                self.isProfile =
                    false

                self.bridgeStatus =
                    "No person"

                self.publishMeasurement()
            }

            return
        }

        let nose =
            Self.point(
                person,
                Self.noseIndex
            )

        let bridge =
            Self.bridgePoint(
                person
            )

        let leftShoulder =
            Self.point(
                person,
                Self.leftShoulderIndex
            )

        let rightShoulder =
            Self.point(
                person,
                Self.rightShoulderIndex
            )

        // -----------------------------------------------------
        // PROFILE
        // -----------------------------------------------------

        let profile:
            Bool

        if
            let nose,
            let bridge
        {

            profile =
                Self.isProfile(
                    person,
                    nose:
                        nose,
                    bridge:
                        bridge
                )

        } else {

            profile =
                false
        }

        // -----------------------------------------------------
        // BRIDGE DEPTH
        // -----------------------------------------------------

        var bridgeFront:
            TrackedPoint?

        var measuredBridge:
            Double?

        var bridgeStatus =
            ""

        if
            let nose,
            let bridge,
            let depth =
                depthData(
                    for:
                        timestampInMilliseconds
                )
        {

            switch Self.findBridgeFront(
                from:
                    bridge,
                toward:
                    nose,
                in:
                    depth
            ) {

            case .found(
                let point,
                let meters
            ):

                bridgeFront =
                    point

                measuredBridge =
                    Double(meters)
                    * 100.0

                lastBridgeOffset =
                    (
                        point.x
                            - bridge.x,
                        timestampInMilliseconds
                    )

                bridgeStatus =
                    "Bridge depth detected"

            case .failed(
                let reason
            ):

                if
                    let previous =
                        lastBridgeOffset,

                    timestampInMilliseconds
                        - previous.timestampMs
                        <= Self.depthHoldMs
                {

                    bridgeFront =
                        TrackedPoint(
                            x:
                                bridge.x
                                + previous.dx,

                            y:
                                bridge.y,

                            visibility:
                                bridge.visibility
                        )

                    if
                        let bridgeFront,
                        let meters =
                            Self.depthMeters(
                                at:
                                    bridgeFront,
                                in:
                                    depth,
                                radius:
                                    2,
                                percentile:
                                    0.25
                            )
                    {

                        measuredBridge =
                            Double(meters)
                            * 100.0
                    }

                    bridgeStatus =
                        "Bridge held: \(reason)"

                } else {

                    bridgeFront =
                        bridge

                    if let meters =
                        Self.depthMeters(
                            at:
                                bridge,
                            in:
                                depth,
                            radius:
                                3,
                            percentile:
                                0.25
                        )
                    {

                        measuredBridge =
                            Double(meters)
                            * 100.0
                    }

                    bridgeStatus =
                        "Bridge fallback: \(reason)"
                }
            }
        }

        let heldBridgeDepth:
            Double?

        if bridge == nil {

            heldBridgeDepth =
                nil

        } else {

            heldBridgeDepth =
                Self.holdDepth(
                    measuredBridge,
                    last:
                        &lastBridgeDepth,
                    timestampMs:
                        timestampInMilliseconds
                )
        }

        // -----------------------------------------------------
        // DEPTH ACCURACY
        // -----------------------------------------------------

        var accuracy =
            ""

        if let depth =
            depthData(
                for:
                    timestampInMilliseconds
            )
        {

            switch depth.depthDataAccuracy {

            case .absolute:

                accuracy =
                    "absolute"

            case .relative:

                accuracy =
                    "relative"

            @unknown default:

                accuracy =
                    "unknown"
            }
        }

        // -----------------------------------------------------
        // PUBLISH
        // -----------------------------------------------------

        DispatchQueue.main.async {

            self.personDetected =
                true

            self.nose =
                nose

            self.bridge =
                bridge

            self.bridgeFront =
                bridgeFront

            self.leftShoulder =
                leftShoulder

            self.rightShoulder =
                rightShoulder

            self.bridgeDepthCM =
                heldBridgeDepth

            self.bridgeStatus =
                bridgeStatus

            self.isProfile =
                profile

            self.depthAccuracy =
                accuracy

            self.publishMeasurement()
        }
    }

    // =========================================================
    // HAND RESULT
    // =========================================================

    func handLandmarker(
        _ handLandmarker:
            HandLandmarker,

        didFinishDetection
            result:
            HandLandmarkerResult?,

        timestampInMilliseconds:
            Int,

        error:
            Error?
    ) {

        if let error {

            print(
                "Hand detection error:",
                error.localizedDescription
            )
        }

        guard
            let hand =
                result?
                    .landmarks
                    .first,

            hand.count >
                Self.thumbTipIndex
        else {

            DispatchQueue.main.async {

                self.thumb =
                    nil

                self.thumbDepthCM =
                    nil

                self.publishMeasurement()
            }

            return
        }

        let thumb =
            Self.point(
                hand,
                Self.thumbTipIndex,
                defaultVisibility:
                    1
            )

        var measuredThumb:
            Double?

        if
            let thumb,
            let depth =
                depthData(
                    for:
                        timestampInMilliseconds
                ),

            let meters =
                Self.depthMeters(
                    at:
                        thumb,
                    in:
                        depth,
                    radius:
                        2,
                    percentile:
                        0.25
                )
        {

            measuredThumb =
                Double(meters)
                * 100.0
        }

        let heldThumb:
            Double?

        if thumb == nil {

            heldThumb =
                nil

        } else {

            heldThumb =
                Self.holdDepth(
                    measuredThumb,
                    last:
                        &lastThumbDepth,
                    timestampMs:
                        timestampInMilliseconds
                )
        }

        DispatchQueue.main.async {

            self.thumb =
                thumb

            self.thumbDepthCM =
                heldThumb

            self.publishMeasurement()
        }
    }

    // =========================================================
    // PUBLISH LATEST MEASUREMENT
    // =========================================================

    private func publishMeasurement() {

        let personDetected =
            self.personDetected

        let thumbDetected =
            self.thumb != nil

        let distance =
            self.bridgeThumbCM

        self.onMeasurement?(
            distance,
            personDetected,
            thumbDetected
        )
    }

    // =========================================================
    // BRIDGE POINT
    // =========================================================

    private static func bridgePoint(
        _ landmarks:
            [NormalizedLandmark]
    ) -> TrackedPoint? {

        guard
            let left =
                point(
                    landmarks,
                    leftEyeInnerIndex
                ),

            let right =
                point(
                    landmarks,
                    rightEyeInnerIndex
                )
        else {
            return nil
        }

        return TrackedPoint(
            x:
                (
                    left.x
                    + right.x
                )
                / 2,

            y:
                (
                    left.y
                    + right.y
                )
                / 2,

            visibility:
                max(
                    left.visibility,
                    right.visibility
                )
        )
    }

    // =========================================================
    // PROFILE DETECTION
    // =========================================================

    private static func isProfile(
        _ landmarks:
            [NormalizedLandmark],

        nose:
            TrackedPoint,

        bridge:
            TrackedPoint
    ) -> Bool {

        guard
            let left =
                point(
                    landmarks,
                    leftEyeInnerIndex
                ),

            let right =
                point(
                    landmarks,
                    rightEyeInnerIndex
                )
        else {
            return false
        }

        let eyeSeparation =
            abs(
                left.x
                    - right.x
            )

        let noseOffset =
            abs(
                nose.x
                    - bridge.x
            )

        return noseOffset
            > eyeSeparation
    }

    // =========================================================
    // HOLD LAST VALID DEPTH
    // =========================================================

    private static func holdDepth(
        _ measured:
            Double?,

        last:
            inout (
                cm: Double,
                timestampMs: Int
            )?,

        timestampMs:
            Int
    ) -> Double? {

        if let measured {

            last =
                (
                    measured,
                    timestampMs
                )

            return measured
        }

        if
            let last,

            timestampMs
                - last.timestampMs
                <= depthHoldMs
        {

            return last.cm
        }

        return nil
    }

    // =========================================================
    // SAMPLE DEPTH
    // =========================================================

    private static func depthMeters(
        at point:
            TrackedPoint,

        in depthData:
            AVDepthData,

        radius:
            Int = 3,

        percentile:
            Double = 0.5
    ) -> Float? {

        withDepthMap(
            depthData
        ) { map in

            let centerX =
                Int(
                    point.x
                    * CGFloat(
                        map.width
                    )
                )

            let centerY =
                Int(
                    point.y
                    * CGFloat(
                        map.height
                    )
                )

            guard
                (0..<map.width)
                    .contains(
                        centerX
                    ),

                (0..<map.height)
                    .contains(
                        centerY
                    )
            else {
                return nil
            }

            var values:
                [Float] = []

            let minY =
                max(
                    0,
                    centerY - radius
                )

            let maxY =
                min(
                    map.height - 1,
                    centerY + radius
                )

            let minX =
                max(
                    0,
                    centerX - radius
                )

            let maxX =
                min(
                    map.width - 1,
                    centerX + radius
                )

            for y in minY...maxY {

                for x in minX...maxX {

                    if let value =
                        map.value(
                            x,
                            y
                        )
                    {

                        values.append(
                            value
                        )
                    }
                }
            }

            guard
                !values.isEmpty
            else {
                return nil
            }

            values.sort()

            let rawIndex =
                Int(
                    Double(
                        values.count
                    )
                    * percentile
                )

            let index =
                min(
                    values.count - 1,
                    max(
                        0,
                        rawIndex
                    )
                )

            return values[
                index
            ]
        }
    }

    // =========================================================
    // BRIDGE SCAN
    // =========================================================

    private enum BridgeScan {

        case found(
            point:
                TrackedPoint,
            meters:
                Float
        )

        case failed(
            String
        )
    }

    private static func findBridgeFront(
        from bridge:
            TrackedPoint,

        toward nose:
            TrackedPoint,

        in depthData:
            AVDepthData
    ) -> BridgeScan {

        guard
            let startDepth =
                depthMeters(
                    at:
                        bridge,
                    in:
                        depthData,
                    radius:
                        3,
                    percentile:
                        0
                )
        else {

            return .failed(
                "no start depth"
            )
        }

        let result:
            BridgeScan? =
            withDepthMap(
                depthData
            ) { map in

                let startX =
                    Int(
                        bridge.x
                        * CGFloat(
                            map.width
                        )
                    )

                let y =
                    Int(
                        bridge.y
                        * CGFloat(
                            map.height
                        )
                    )

                guard
                    (0..<map.width)
                        .contains(
                            startX
                        ),

                    (1..<(map.height - 1))
                        .contains(
                            y
                        )
                else {

                    return .failed(
                        "start outside image"
                    )
                }

                let step =
                    nose.x >= bridge.x
                    ? 1
                    : -1

                func columnDepth(
                    _ x: Int
                ) -> Float? {

                    [
                        map.value(
                            x,
                            y - 1
                        ),

                        map.value(
                            x,
                            y
                        ),

                        map.value(
                            x,
                            y + 1
                        )
                    ]
                    .compactMap {
                        $0
                    }
                    .min()
                }

                let backgroundJump:
                    Float = 0.10

                let maxSteps =
                    max(
                        20,
                        map.width
                            * 80
                            / 480
                    )

                let edgeRun =
                    3

                var faceDepths:
                    [Float] = [
                        startDepth
                    ]

                var lastFaceX =
                    startX

                var backgroundCount =
                    0

                var x =
                    startX

                for _ in 0..<maxSteps {

                    x += step

                    guard
                        (0..<map.width)
                            .contains(
                                x
                            )
                    else {

                        return .failed(
                            "reached image edge"
                        )
                    }

                    if
                        let depth =
                            columnDepth(
                                x
                            ),

                        depth
                            <
                            startDepth
                            + backgroundJump
                    {

                        lastFaceX =
                            x

                        faceDepths.append(
                            depth
                        )

                        backgroundCount =
                            0

                    } else {

                        backgroundCount +=
                            1

                        if
                            backgroundCount
                                >= edgeRun
                        {

                            let recent =
                                faceDepths
                                    .suffix(
                                        5
                                    )
                                    .sorted()

                            let meters =
                                recent[
                                    recent.count
                                        / 2
                                ]

                            let point =
                                TrackedPoint(
                                    x:
                                        (
                                            CGFloat(
                                                lastFaceX
                                            )
                                            + 0.5
                                        )
                                        /
                                        CGFloat(
                                            map.width
                                        ),

                                    y:
                                        bridge.y,

                                    visibility:
                                        bridge.visibility
                                )

                            return .found(
                                point:
                                    point,
                                meters:
                                    meters
                            )
                        }
                    }
                }

                return .failed(
                    "no edge within \(maxSteps) px"
                )
            }

        return result
            ?? .failed(
                "depth map unreadable"
            )
    }

    // =========================================================
    // DEPTH DEBUG
    // =========================================================

    static func validDepthFraction(
        _ depthData:
            AVDepthData
    ) -> Double? {

        withDepthMap(
            depthData
        ) { map in

            var valid =
                0

            for y in 0..<map.height {

                for x in 0..<map.width {

                    if
                        map.value(
                            x,
                            y
                        )
                        != nil
                    {

                        valid +=
                            1
                    }
                }
            }

            return Double(
                valid
            )
            /
            Double(
                map.width
                    * map.height
            )
        }
    }

    // =========================================================
    // DEPTH MAP
    // =========================================================

    private struct DepthMap {

        let base:
            UnsafeMutableRawPointer

        let width:
            Int

        let height:
            Int

        let bytesPerRow:
            Int

        func value(
            _ x:
                Int,
            _ y:
                Int
        ) -> Float? {

            guard
                (0..<width)
                    .contains(
                        x
                    ),

                (0..<height)
                    .contains(
                        y
                    )
            else {
                return nil
            }

            let value =
                base
                    .advanced(
                        by:
                            y
                            * bytesPerRow
                    )
                    .assumingMemoryBound(
                        to:
                            Float32.self
                    )[x]

            guard
                value.isFinite,
                value > 0
            else {
                return nil
            }

            return value
        }
    }

    // =========================================================
    // READ DEPTH MAP
    // =========================================================

    private static func withDepthMap<T>(
        _ depthData:
            AVDepthData,

        _ body:
            (
                DepthMap
            ) -> T?
    ) -> T? {

        let depth:
            AVDepthData

        if
            depthData.depthDataType
                ==
                kCVPixelFormatType_DepthFloat32
        {

            depth =
                depthData

        } else {

            depth =
                depthData.converting(
                    toDepthDataType:
                        kCVPixelFormatType_DepthFloat32
                )
        }

        let pixelBuffer =
            depth.depthDataMap

        CVPixelBufferLockBaseAddress(
            pixelBuffer,
            .readOnly
        )

        defer {

            CVPixelBufferUnlockBaseAddress(
                pixelBuffer,
                .readOnly
            )
        }

        guard
            let base =
                CVPixelBufferGetBaseAddress(
                    pixelBuffer
                )
        else {
            return nil
        }

        return body(
            DepthMap(
                base:
                    base,

                width:
                    CVPixelBufferGetWidth(
                        pixelBuffer
                    ),

                height:
                    CVPixelBufferGetHeight(
                        pixelBuffer
                    ),

                bytesPerRow:
                    CVPixelBufferGetBytesPerRow(
                        pixelBuffer
                    )
            )
        )
    }

    // =========================================================
    // LANDMARK -> TRACKED POINT
    // =========================================================

    private static func point(
        _ landmarks:
            [NormalizedLandmark],

        _ index:
            Int,

        defaultVisibility:
            Float = 0
    ) -> TrackedPoint? {

        guard
            index <
                landmarks.count
        else {
            return nil
        }

        let landmark =
            landmarks[
                index
            ]

        return TrackedPoint(
            x:
                CGFloat(
                    landmark.x
                ),

            y:
                CGFloat(
                    landmark.y
                ),

            visibility:
                landmark
                    .visibility?
                    .floatValue
                ??
                defaultVisibility
        )
    }
}
