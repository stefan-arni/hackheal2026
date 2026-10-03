// BessTestView.swift
// posecam iPhone side: three BESS test buttons + live status + score table.
//
// Drop this file into your Xcode project (iOS 16+). It contains:
//   - PosecamClient: WebSocket client for the pose server (port 8765). Sends
//     test commands, and frames if you call sendFrame(_:) from your camera code.
//   - BessTestView: the screen with a button for each of the three tests.
//     (Eye tracking is off for now: eyes are not checked or scored.)
//
// Usage:
//   @StateObject private var posecam = PosecamClient()
//   ...
//   BessTestView(client: posecam)
//   // from your camera pipeline, for each frame:
//   posecam.sendFrame(jpegData)
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
}

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
    @Published var errors = 0
    @Published var activeErrors: [String] = []
    @Published var log: [BessLogEntry] = []
    @Published var warnings: [String] = []
    @Published var scores: [BessStance: Int] = [:]
    @Published var total = 0
    @Published var complete = false
    @Published var lastResult: String?
    @Published var notice: String?

    private var task: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)

    static let errorLabels: [String: String] = [
        "hands_off_hips": "Hands off hips",
        "step_stumble_fall": "Step / stumble / fall",
        "hip_angle": "Hip > 30°",
        "foot_lift": "Heel / forefoot lifted",
        "out_of_position": "Out of position > 5 s",
    ]

    func connect(to url: URL) {
        disconnect()
        let t = session.webSocketTask(with: url)
        task = t
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

    func start(_ stance: BessStance, nondominant: String) {
        lastResult = nil
        send(["type": "bess_start", "stance": stance.rawValue, "nondominant": nondominant])
    }

    func cancel() { send(["type": "bess_cancel"]) }
    func resetScores() { lastResult = nil; send(["type": "bess_reset"]) }

    private func send(_ object: [String: Any]) {
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

        switch type {
        case "pose":
            if let bess = json["bess"] as? [String: Any] { apply(bess) }
        case "ack":
            applySession(json)
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
            notice = json["error"] as? String
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
        errors = b["errors"] as? Int ?? 0
        activeErrors = (b["active"] as? [String] ?? []).map { Self.errorLabels[$0] ?? $0 }
        warnings = b["warnings"] as? [String] ?? []
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
        for st in BessStance.allCases {
            if let v = raw[st.rawValue] as? Int { out[st] = v }
        }
        scores = out
        total = s["total"] as? Int ?? 0
        complete = s["complete"] as? Bool ?? false
    }
}

// MARK: - Screen

struct BessTestView: View {
    @ObservedObject var client: PosecamClient
    @AppStorage("posecamHost") private var host = "192.168.1.20"
    @AppStorage("bessNondominant") private var nondominant = "left"

    private var testInProgress: Bool { client.phase != "idle" }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                connectionBar

                Text("Balance Test (BESS)").font(.title2.bold())
                Text("20 seconds each, hands on hips.")
                    .font(.subheadline).foregroundStyle(.secondary)

                Picker("Non-dominant leg", selection: $nondominant) {
                    Text("Non-dominant: Left").tag("left")
                    Text("Non-dominant: Right").tag("right")
                }
                .pickerStyle(.segmented)
                .disabled(testInProgress)

                ForEach(BessStance.allCases) { stance in
                    testButton(stance)
                }

                status
                scoreTable

                if let notice = client.notice {
                    Text(notice).font(.footnote).foregroundStyle(.red)
                }
            }
            .padding()
        }
    }

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

    private func testButton(_ stance: BessStance) -> some View {
        let isActive = testInProgress && client.stance == stance
        return Button {
            client.start(stance, nondominant: nondominant)
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(stance.title).font(.headline)
                    Text(stance.instructions).font(.caption).opacity(0.85)
                }
                Spacer()
                if let score = client.scores[stance] {
                    Text("\(score)").font(.title2.monospacedDigit().bold())
                }
            }
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.borderedProminent)
        .tint(isActive ? .orange : .accentColor)
        .disabled(!client.isConnected || testInProgress)
    }

    @ViewBuilder
    private var status: some View {
        switch client.phase {
        case "countdown":
            VStack(alignment: .leading, spacing: 8) {
                Text("\(client.stance?.title ?? ""): get in position")
                    .font(.headline)
                Text("Hands on hips.").foregroundStyle(.secondary)
                if client.waitingForView {
                    Text("Waiting to see the whole body. Step back so feet are in frame.")
                        .foregroundStyle(.red)
                } else {
                    Text("\(Int(client.countdownLeft.rounded(.up)))")
                        .font(.system(size: 64, weight: .bold, design: .rounded))
                        .frame(maxWidth: .infinity)
                }
                Button("Cancel", role: .destructive) { client.cancel() }
            }
        case "running":
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(client.stance?.title ?? "").font(.headline)
                    Spacer()
                    Text(String(format: "%.1f s", client.timeLeft))
                        .font(.title3.monospacedDigit().bold())
                }
                ProgressView(value: max(0, 20 - client.timeLeft), total: 20)
                Text("Errors: \(client.errors)")
                    .font(.title.bold())
                    .foregroundStyle(client.errors == 0 ? Color.green : Color.red)
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
        default:
            if let result = client.lastResult {
                Text(result).font(.headline)
            } else {
                Text("Tap a test to start.").foregroundStyle(.secondary)
            }
        }
    }

    private var scoreTable: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Score (errors)").font(.headline)
                Spacer()
                Button("Reset") { client.resetScores() }
                    .font(.footnote)
                    .disabled(!client.isConnected || testInProgress)
            }
            ForEach(BessStance.allCases) { stance in
                HStack {
                    Text(stance.title)
                    Spacer()
                    Text(client.scores[stance].map(String.init) ?? "–").monospacedDigit()
                }
            }
            Divider()
            HStack {
                Text(client.complete ? "Total" : "Total (so far)").bold()
                Spacer()
                Text("\(client.total)").bold().monospacedDigit()
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

#Preview {
    BessTestView(client: PosecamClient())
}
