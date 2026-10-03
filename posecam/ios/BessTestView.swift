// BessTestView.swift
// posecam iPhone side: the Modified BESS balance test.
//
// Three stances on a firm surface (length set by the server, 10 s by default),
// hands on hips, eyes closed:
//   Feet together, Tandem (non-dominant foot in back), Single leg (on the
//   non-dominant leg). One point for each error: hands off hips, eyes opened
//   (marked by the examiner), step / stumble / fall, hip past 30° of flexion or
//   abduction, heel or forefoot lifted, out of position for more than 5 s.
//   Output: the sum of all points, broken down by stance.
//
// Screen: live video with the skeleton, the three test steps, the live error
// count, a summary with reference ranges, and the parameter table.
//
// Also contains PosecamClient: WebSocket client for the pose server (port 8765).
// Sends test commands and camera frames (with depth, from DepthCamera), and
// passes every server message to `listeners`.
//
// Info.plist (plain ws:// to a laptop on the same Wi-Fi):
//   NSAppTransportSecurity > NSAllowsLocalNetworking = YES
//   NSLocalNetworkUsageDescription = "Connects to the posecam server on your network."

import Foundation
import SwiftUI

// MARK: - Model

enum BessStance: String, CaseIterable, Identifiable {
    case double, tandem, single

    var id: String { rawValue }

    var title: String {
        switch self {
        case .double: return "Feet Together"
        case .tandem: return "Tandem"
        case .single: return "Single Leg"
        }
    }

    var instructions: String {
        switch self {
        case .double: return "Feet side by side, touching"
        case .tandem: return "One foot in front, non-dominant foot in back"
        case .single: return "Stand on the non-dominant leg"
        }
    }

    var icon: String {
        switch self {
        case .double: return "figure.stand"
        case .tandem: return "figure.walk"
        case .single: return "figure.cooldown"
        }
    }

    /// Approximate errors for healthy adults on a firm surface. A reference for
    /// the demo, not a validated norm: check against your clinical source.
    var reference: ClosedRange<Double> {
        switch self {
        case .double: return 0...1
        case .tandem: return 0...2
        case .single: return 0...4
        }
    }
}

let bessTotalReference: ClosedRange<Double> = 0...6

struct BessLogEntry: Identifiable {
    let id = UUID()
    let time: Double
    let label: String
    let counted: Bool
}

// MARK: - WebSocket client

@MainActor
final class PosecamClient: ObservableObject {
    @Published var isConnected = false
    @Published var phase = "idle"                 // idle | countdown | running
    @Published var stance: BessStance?
    @Published var countdownLeft: Double = 0
    @Published var waitingForView = false
    @Published var timeLeft: Double = 0
    @Published var duration: Double?             // stance length from the server (duration_s)
    @Published var errors = 0
    @Published var activeErrors: [String] = []
    @Published var log: [BessLogEntry] = []
    @Published var warnings: [String] = []
    @Published var hipAngles: [String: Double] = [:]
    @Published var liveSwayCm: Double?
    @Published var scores: [BessStance: Int] = [:]
    @Published var byStance: [BessStance: [String: Int]] = [:]
    @Published var swayByStance: [BessStance: [String: Any]] = [:]
    @Published var coverage: [BessStance: Double] = [:]
    @Published var errorTypes: [String] = []
    @Published var total = 0
    @Published var complete = false
    @Published var lastResult: String?
    @Published var notice: String?
    @Published var errorPulse = 0                 // +1 on every counted error (vibration)
    @Published var landmarks: [Int: CGPoint] = [:]
    @Published var tracking: Double?              // mean landmark visibility, 0-1
    @Published var torsoDistance: Double?
    @Published var feet: String?                  // both_down | left_up | right_up | not_visible

    /// Every message from the server, for other screens (e.g. the sway tests).
    var listeners: [([String: Any]) -> Void] = []

    private var task: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private var framesInFlight = 0

    static let errorOrder = ["hands_off_hips", "eyes_open", "step_stumble_fall", "hip_angle",
                             "foot_lift", "out_of_position"]
    static let errorLabels: [String: String] = [
        "hands_off_hips": "Hands off hips",
        "eyes_open": "Eyes opened",
        "step_stumble_fall": "Step / stumble / fall",
        "hip_angle": "Hip > 30°",
        "foot_lift": "Heel / forefoot lifted",
        "out_of_position": "Out of position > 5 s",
    ]
    static let shownLandmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

    func connect(to url: URL) {
        disconnect()
        let t = session.webSocketTask(with: url)
        task = t
        framesInFlight = 0
        t.resume()
        isConnected = true
        notice = nil
        receive()
        send(["type": "bess_status"])
    }

    func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        isConnected = false
    }

    /// Send one camera frame (JPEG). Call this from your capture pipeline.
    func sendFrame(_ jpeg: Data) {
        task?.send(.data(jpeg)) { _ in }
    }

    /// Send a frame built by `frameMessage(_:)`. Skips it while two frames are
    /// still on their way, so a slow network never builds up a backlog.
    func sendFrameMessage(_ text: String) {
        guard let task, framesInFlight < 2 else { return }
        framesInFlight += 1
        task.send(.string(text)) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.framesInFlight = max(0, self.framesInFlight - 1)
            }
        }
    }

    /// JSON frame message (JPEG + optional depth map), built off the main thread.
    nonisolated static func frameMessage(_ f: DepthFrame) -> String? {
        var msg: [String: Any] = [
            "type": "frame", "frame_id": f.index, "timestamp_ms": f.timestampMs,
            "rotate": f.rotate, "camera": f.camera, "image": f.jpeg.base64EncodedString(),
        ]
        if let depth = f.depthMillimetres, f.depthWidth > 0, f.intrinsics.count == 4 {
            msg["depth"] = depth.base64EncodedString()
            msg["depth_format"] = "uint16_mm"
            msg["depth_size"] = [f.depthWidth, f.depthHeight]
            msg["intrinsics"] = f.intrinsics
        }
        guard let data = try? JSONSerialization.data(withJSONObject: msg) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func start(_ stance: BessStance, nondominant: String) {
        lastResult = nil
        send(["type": "bess_start", "stance": stance.rawValue, "nondominant": nondominant])
    }

    func cancel() { send(["type": "bess_cancel"]) }
    func resetScores() { lastResult = nil; send(["type": "bess_reset"]) }
    /// The examiner saw the eyes open (the camera can't see that at this distance).
    func markEyesOpen() { send(["type": "bess_mark", "error": "eyes_open"]) }

    func send(_ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(text)) { [weak self] error in
            guard let error else { return }
            Task { @MainActor in self?.notice = error.localizedDescription }
        }
    }

    private func receive() {
        task?.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .failure(let error):
                    self.isConnected = false
                    self.notice = "Disconnected: \(error.localizedDescription)"
                case .success(let message):
                    switch message {
                    case .string(let text): self.handle(text.data(using: .utf8))
                    case .data(let data): self.handle(data)
                    @unknown default: break
                    }
                    self.receive()
                }
            }
        }
    }

    private func handle(_ data: Data?) {
        guard let data,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }
        listeners.forEach { $0(json) }

        switch type {
        case "pose":
            if let bess = json["bess"] as? [String: Any] { apply(bess) }
            applyLandmarks(json)
            feet = (json["feet"] as? [String: Any])?["state"] as? String
        case "ack":
            if (json["command"] as? String ?? "").hasPrefix("bess_") { applySession(json) }
        case "event":
            if json["kind"] as? String == "bess_error", json["counted"] as? Bool == true {
                errorPulse += 1
            }
        case "result":
            if json["kind"] as? String == "bess_done" {
                let label = json["label"] as? String ?? "Test"
                let n = json["errors"] as? Int ?? 0
                lastResult = "\(label): \(n) error\(n == 1 ? "" : "s")"
                if let s = json["session"] as? [String: Any] { applySession(s) }
            }
        case "status":
            if json["kind"] as? String == "bess_failed" {
                notice = json["reason"] as? String
            }
        case "error":
            if (json["command"] as? String ?? "").hasPrefix("bess_") || json["command"] == nil {
                notice = json["error"] as? String
            }
        default:
            break   // alerts/events for other features, pong, ...
        }
    }

    private func apply(_ b: [String: Any]) {
        phase = b["phase"] as? String ?? "idle"
        stance = (b["stance"] as? String).flatMap(BessStance.init(rawValue:))
        countdownLeft = b["countdown_left"] as? Double ?? 0
        waitingForView = b["waiting_for_view"] as? Bool ?? false
        timeLeft = b["time_left"] as? Double ?? 0
        if let d = b["duration_s"] as? Double { duration = d }
        errors = b["errors"] as? Int ?? 0
        activeErrors = (b["active"] as? [String] ?? []).map { Self.errorLabels[$0] ?? $0 }
        warnings = b["warnings"] as? [String] ?? []
        hipAngles = b["hip_angles"] as? [String: Double] ?? [:]
        liveSwayCm = b["sway_cm"] as? Double
        if let entries = b["log"] as? [[String: Any]] {
            log = entries.map {
                BessLogEntry(time: $0["t"] as? Double ?? 0,
                             label: $0["label"] as? String ?? "",
                             counted: $0["counted"] as? Bool ?? false)
            }
        } else if phase != "running" {
            log = []
        }
        if let s = b["session"] as? [String: Any] { applySession(s) }
    }

    private func applySession(_ s: [String: Any]) {
        guard let raw = s["scores"] as? [String: Any] else { return }
        var out: [BessStance: Int] = [:]
        var types: [BessStance: [String: Int]] = [:]
        var cov: [BessStance: Double] = [:]
        var sway: [BessStance: [String: Any]] = [:]
        let byStanceRaw = s["by_stance"] as? [String: Any] ?? [:]
        let covRaw = s["coverage"] as? [String: Any] ?? [:]
        let swayRaw = s["sway"] as? [String: Any] ?? [:]
        for st in BessStance.allCases {
            if let v = raw[st.rawValue] as? Int { out[st] = v }
            if let t = byStanceRaw[st.rawValue] as? [String: Int] { types[st] = t }
            if let c = covRaw[st.rawValue] as? Double { cov[st] = c }
            if let m = swayRaw[st.rawValue] as? [String: Any] { sway[st] = m }
        }
        scores = out
        byStance = types
        coverage = cov
        swayByStance = sway
        if let et = s["error_types"] as? [String] { errorTypes = et }
        total = s["total"] as? Int ?? 0
        complete = s["complete"] as? Bool ?? false
    }

    private func applyLandmarks(_ m: [String: Any]) {
        if let d = m["depth"] as? [String: Any] {
            torsoDistance = (d["torso_m"] as? [Double]).flatMap { $0.count == 3 ? $0[2] : nil }
        } else {
            torsoDistance = nil
        }
        guard let lms = m["landmarks"] as? [[String: Any]], lms.count == 33 else {
            landmarks = [:]
            tracking = nil
            return
        }
        var out: [Int: CGPoint] = [:]
        var vis: [Double] = []
        for i in Self.shownLandmarks {
            let lm = lms[i]
            let v = lm["visibility"] as? Double ?? 0
            vis.append(v)
            guard v > 0.3, let x = lm["x"] as? Double, let y = lm["y"] as? Double else { continue }
            out[i] = CGPoint(x: x, y: y)
        }
        landmarks = out
        tracking = vis.reduce(0, +) / Double(max(vis.count, 1))
    }
}

// MARK: - Small views

/// Reference-range bar: grey track, green "typical" zone, black marker at the value.
struct RangeBar: View {
    let value: Double?
    let range: ClosedRange<Double>
    let scaleMax: Double

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let x = { (v: Double) in CGFloat(min(max(v / scaleMax, 0), 1)) * w }
            ZStack(alignment: .leading) {
                Capsule().fill(Color.gray.opacity(0.2))
                Capsule().fill(Color.green.opacity(0.45))
                    .frame(width: max(4, x(range.upperBound) - x(range.lowerBound)))
                    .offset(x: x(range.lowerBound))
                if let v = value {
                    RoundedRectangle(cornerRadius: 1).fill(Color.primary)
                        .frame(width: 3, height: 14)
                        .offset(x: min(max(x(v) - 1.5, 0), w - 3))
                }
            }
        }
        .frame(height: 14)
    }
}

struct MetricTile: View {
    let title: String
    let value: String
    let note: String
    let barValue: Double?
    let range: ClosedRange<Double>
    let scaleMax: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.bold().monospacedDigit())
            Text(note).font(.caption2).foregroundStyle(.secondary)
            RangeBar(value: barValue, range: range, scaleMax: scaleMax)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Screen

struct BessTestView: View {
    @ObservedObject var client: PosecamClient
    @ObservedObject var camera: DepthCamera
    @AppStorage("posecamHost") private var host = "192.168.1.20"
    @AppStorage("bessNondominant") private var nondominant = "left"

    private var testInProgress: Bool { client.phase != "idle" }
    /// e.g. "10 s", from the server's stance length; nil until the server has said.
    private var durationText: String? { client.duration.map { "\(Int($0.rounded())) s" } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                connectionBar
                videoCard
                Picker("Non-dominant leg", selection: $nondominant) {
                    Text("Non-dominant: Left").tag("left")
                    Text("Non-dominant: Right").tag("right")
                }
                .pickerStyle(.segmented)
                .disabled(testInProgress)

                testSteps
                livePanel
                summaryCard
                parametersCard

                if let notice = client.notice ?? camera.error {
                    Text(notice).font(.footnote).foregroundStyle(.red)
                }
            }
            .padding()
        }
        .sensoryFeedback(.error, trigger: client.errorPulse)
    }

    // MARK: connection + video

    private var connectionBar: some View {
        HStack {
            TextField("Laptop IP", text: $host)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numbersAndPunctuation)
                .autocorrectionDisabled()
            Button(client.isConnected ? "Disconnect" : "Connect") {
                if client.isConnected {
                    client.disconnect()
                } else if let url = URL(string: "ws://\(host):8765") {
                    client.connect(to: url)
                }
            }
            .buttonStyle(.bordered)
            Circle()
                .fill(client.isConnected ? Color.green : Color.gray)
                .frame(width: 10, height: 10)
        }
    }

    private var videoCard: some View {
        ZStack {
            CameraPreview(session: camera.session, running: camera.running)
            SkeletonOverlay(points: client.landmarks, mirrored: false)
            if !camera.running {
                Button("Start Camera") { camera.start(.lidar) }
                    .buttonStyle(.borderedProminent)
            }
            if client.phase == "countdown" && !client.waitingForView {
                Text("\(Int(client.countdownLeft.rounded(.up)))")
                    .font(.system(size: 90, weight: .bold, design: .rounded))
                    .foregroundStyle(.white).shadow(radius: 6)
            }
        }
        .aspectRatio(3.0 / 4.0, contentMode: .fit)
        .frame(maxWidth: .infinity, maxHeight: 380)
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(alignment: .topLeading) { statusBadge.padding(8) }
        .overlay(alignment: .topTrailing) {
            if let st = client.stance, testInProgress {
                Text(st.title).font(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Color.blue, in: Capsule()).foregroundStyle(.white).padding(8)
            }
        }
        .overlay(alignment: .bottomTrailing) { feetChip.padding(8) }
        .overlay(alignment: .bottomLeading) {
            if camera.running, let tr = client.tracking {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Pose tracking")
                    Text(String(format: "Confidence %.0f%%", tr * 100)).bold()
                    if let z = client.torsoDistance { Text(String(format: "Depth %.1f m", z)) }
                }
                .font(.caption2).foregroundStyle(.white)
                .padding(6).background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
                .padding(8)
            }
        }
    }

    /// Each foot up or down, green when it matches the stance being tested.
    @ViewBuilder
    private var feetChip: some View {
        if camera.running, let feet = client.feet {
            let text: String = switch feet {
            case "both_down": "Both feet down"
            case "left_up": "Left foot up"
            case "right_up": "Right foot up"
            default: "Feet not visible"
            }
            let expected: String? = switch client.stance {
            case .double?, .tandem?: "both_down"
            case .single?: nondominant == "left" ? "right_up" : "left_up"
            case nil: nil
            }
            let color: Color = !testInProgress || expected == nil ? .blue
                : (feet == expected ? .green : .red)
            Label(text, systemImage: "shoeprints.fill")
                .font(.caption.bold())
                .foregroundStyle(.white)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(color.opacity(0.85), in: Capsule())
        }
    }

    private var statusBadge: some View {
        let text: String
        let dot: Color
        switch client.phase {
        case "running":
            let total = client.duration ?? client.timeLeft
            let elapsed = max(0, total - client.timeLeft)
            text = String(format: "Running  0:%02d / 0:%02d", Int(elapsed), Int(total.rounded()))
            dot = .red
        case "countdown":
            text = client.waitingForView ? "Step back: feet not in view" : "Get in position"
            dot = .orange
        default:
            text = camera.running ? (client.isConnected ? "Ready" : "Not connected") : "Camera off"
            dot = client.isConnected && camera.running ? .green : .gray
        }
        return HStack(spacing: 6) {
            Circle().fill(dot).frame(width: 9, height: 9)
            Text(text).font(.caption.bold().monospacedDigit())
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.black.opacity(0.65), in: Capsule())
    }

    // MARK: test steps

    private var testSteps: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Test Steps").font(.headline)
            HStack(spacing: 8) {
                ForEach(BessStance.allCases) { stepCard($0) }
            }
            Text(client.stance.map { $0.instructions }
                 ?? "Tap a step to start it. Hands on hips, eyes closed\(durationText.map { ", \($0)" } ?? "").")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func stepCard(_ st: BessStance) -> some View {
        let running = testInProgress && client.stance == st
        let score = client.scores[st]
        let status = running ? (client.phase == "countdown" ? "Get ready" : "Running")
                             : (score != nil ? "Completed" : "Pending")
        let statusColor: Color = running ? .blue : (score != nil ? .green : .secondary)
        return Button {
            client.start(st, nondominant: nondominant)
        } label: {
            VStack(spacing: 6) {
                Image(systemName: st.icon).font(.system(size: 28))
                    .frame(height: 34)
                Text(st.title).font(.caption.bold()).lineLimit(1).minimumScaleFactor(0.8)
                Text(status).font(.caption2).foregroundStyle(statusColor)
                Text(score.map { "\($0) error\($0 == 1 ? "" : "s")" } ?? durationText ?? "–")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(score.map { $0 == 0 ? Color.green : Color.red } ?? Color.secondary)
            }
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12)
                .stroke(running ? Color.blue : Color.clear, lineWidth: 2))
        }
        .buttonStyle(.plain)
        .disabled(!client.isConnected || !camera.running || testInProgress)
    }

    // MARK: live

    @ViewBuilder
    private var livePanel: some View {
        switch client.phase {
        case "running":
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(client.stance?.title ?? "").font(.headline)
                    Spacer()
                    Text(String(format: "%.1f s", client.timeLeft)).font(.title3.monospacedDigit().bold())
                }
                let total = max(client.duration ?? client.timeLeft, 1)
                ProgressView(value: max(0, total - client.timeLeft), total: total)
                HStack(alignment: .firstTextBaseline) {
                    Text("Errors: \(client.errors)")
                        .font(.title.bold())
                        .foregroundStyle(client.errors == 0 ? Color.green : Color.red)
                    Spacer()
                    if client.errorTypes.contains("eyes_open") {
                        Button("Eyes opened +1") { client.markEyesOpen() }
                            .buttonStyle(.borderedProminent).tint(.red)
                    }
                }
                ForEach(client.activeErrors, id: \.self) { e in
                    Text("Now: \(e)").foregroundStyle(.red)
                }
                ForEach(client.log.reversed()) { entry in
                    HStack {
                        Text(String(format: "%.1fs", entry.time)).monospacedDigit()
                        Text(entry.counted ? "+1" : "0").bold()
                        Text(entry.label)
                    }
                    .font(.footnote)
                    .foregroundStyle(entry.counted ? Color.primary : Color.secondary)
                }
                ForEach(client.warnings, id: \.self) { w in
                    Label(w, systemImage: "exclamationmark.triangle").font(.footnote)
                        .foregroundStyle(.orange)
                }
                Button("Cancel", role: .destructive) { client.cancel() }
            }
            .padding()
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
        case "countdown":
            HStack {
                Text("\(client.stance?.title ?? ""): get in position, hands on hips, eyes closed")
                    .font(.subheadline)
                Spacer()
                Button("Cancel", role: .destructive) { client.cancel() }
            }
        default:
            if let result = client.lastResult {
                Text(result).font(.headline)
            }
        }
    }

    // MARK: summary

    private var summaryCard: some View {
        let chip: (String, Color) = !client.complete
            ? ("In progress", .gray)
            : (bessTotalReference.contains(Double(client.total)) ? ("Within typical range", .green)
                                                                 : ("Above typical range", .orange))
        let hips = client.hipAngles.values.max()
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("BESS Summary").font(.headline)
                Spacer()
                Text(chip.0).font(.caption.bold())
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(chip.1.opacity(0.18), in: Capsule())
                    .foregroundStyle(chip.1)
            }
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                MetricTile(title: client.complete ? "Total errors" : "Total (so far)",
                           value: "\(client.total)",
                           note: "(ref. \(Int(bessTotalReference.lowerBound))–\(Int(bessTotalReference.upperBound)))",
                           barValue: client.scores.isEmpty ? nil : Double(client.total),
                           range: bessTotalReference, scaleMax: 30)
                ForEach(BessStance.allCases) { st in
                    MetricTile(title: st.title,
                               value: client.scores[st].map(String.init) ?? "–",
                               note: "(ref. \(Int(st.reference.lowerBound))–\(Int(st.reference.upperBound)))",
                               barValue: client.scores[st].map(Double.init),
                               range: st.reference, scaleMax: 10)
                }
                if client.phase == "running" {
                    MetricTile(title: "Hip angle (live)",
                               value: hips.map { String(format: "%.0f°", $0) } ?? "–",
                               note: "(error > 30°)",
                               barValue: hips, range: 0...30, scaleMax: 60)
                    MetricTile(title: "Sway (live)",
                               value: client.liveSwayCm.map { String(format: "%.1f cm", $0) } ?? "–",
                               note: "from start position",
                               barValue: client.liveSwayCm, range: 0...5, scaleMax: 15)
                }
            }
            Text("Reference ranges are approximate, for the demo.")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: parameters table

    private var parametersCard: some View {
        let types = PosecamClient.errorOrder.filter { client.errorTypes.isEmpty || client.errorTypes.contains($0) }
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Balance Parameters").font(.headline)
                Spacer()
                Button("Reset") { client.resetScores() }
                    .font(.footnote)
                    .disabled(!client.isConnected || testInProgress)
            }
            Grid(alignment: .leading, horizontalSpacing: 8, verticalSpacing: 6) {
                GridRow {
                    Text("Error").bold()
                    Text("Feet").bold().gridColumnAlignment(.trailing)
                    Text("Tand.").bold().gridColumnAlignment(.trailing)
                    Text("Single").bold().gridColumnAlignment(.trailing)
                    Text("Total").bold().gridColumnAlignment(.trailing)
                }
                Divider().gridCellUnsizedAxes(.horizontal)
                ForEach(types, id: \.self) { k in
                    GridRow {
                        Text(PosecamClient.errorLabels[k] ?? k).lineLimit(1).minimumScaleFactor(0.75)
                        ForEach(BessStance.allCases) { st in cell(client.byStance[st]?[k]) }
                        cell(rowTotal(k), bold: true)
                    }
                }
                Divider().gridCellUnsizedAxes(.horizontal)
                GridRow {
                    Text("Errors").bold()
                    ForEach(BessStance.allCases) { st in cell(client.scores[st], bold: true) }
                    cell(client.scores.isEmpty ? nil : client.total, bold: true)
                }
                Divider().gridCellUnsizedAxes(.horizontal)
                GridRow {
                    Text("Sway (cm/s)")
                    ForEach(BessStance.allCases) { st in
                        text(client.swayByStance[st]?["mean_velocity_cm_s"] as? Double, "%.1f")
                    }
                    Text("")
                }
                GridRow {
                    Text("Sway area (cm²)")
                    ForEach(BessStance.allCases) { st in
                        text(client.swayByStance[st]?["area_95_cm2"] as? Double, "%.0f")
                    }
                    Text("")
                }
                GridRow {
                    Text("Tracked")
                    ForEach(BessStance.allCases) { st in
                        text(client.coverage[st].map { $0 * 100 }, "%.0f%%")
                    }
                    Text("")
                }
            }
            .font(.footnote)
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func rowTotal(_ k: String) -> Int? {
        let vals = BessStance.allCases.compactMap { client.byStance[$0]?[k] }
        return vals.isEmpty ? nil : vals.reduce(0, +)
    }

    private func cell(_ v: Int?, bold: Bool = false) -> some View {
        Text(v.map(String.init) ?? "–")
            .fontWeight(bold ? .bold : .regular)
            .monospacedDigit()
            .foregroundStyle(v.map { $0 > 0 ? Color.red : Color.primary } ?? Color.secondary)
    }

    private func text(_ v: Double?, _ format: String) -> some View {
        Text(v.map { String(format: format, $0) } ?? "–").monospacedDigit()
            .foregroundStyle(v == nil ? Color.secondary : Color.primary)
    }
}

#Preview {
    BessTestView(client: PosecamClient(), camera: DepthCamera())
}
