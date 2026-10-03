import SwiftUI

// Full-screen instructions for the patient during a BESS stance.
//
// The patient stands 2-3 m from the phone, so everything is very large. It
// shows only which stance, how to stand and what to do now ("Get ready",
// "Hold still", "Done"): no scores, errors or numbers. The doctor starts each
// stance; this screen just follows the pose server's BESS state.

struct BalanceInstructionView: View {
    @ObservedObject var streamer: VideoStreamClient

    private static let order = ["double", "tandem", "single"]

    private var stance: String? { streamer.bessStance ?? streamer.bessJustFinished }

    var body: some View {
        VStack(spacing: 28) {
            steps
                .padding(.top, 24)

            Spacer(minLength: 0)

            if let stance {
                Text(Self.title(stance).uppercased())
                    .font(.system(size: 54, weight: .heavy, design: .rounded))
                    .minimumScaleFactor(0.5)
                    .lineLimit(1)
                    .foregroundStyle(.white)

                Text(instructions(stance))
                    .font(.system(size: 30, weight: .semibold, design: .rounded))
                    .multilineTextAlignment(.center)
                    .minimumScaleFactor(0.6)
                    .foregroundStyle(.white.opacity(0.9))
            }

            Spacer(minLength: 0)

            Text(phaseText)
                .font(.system(size: 46, weight: .bold, design: .rounded))
                .minimumScaleFactor(0.5)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white)
                .padding(.horizontal, 28)
                .padding(.vertical, 18)
                .frame(maxWidth: .infinity)
                .background(phaseColor, in: RoundedRectangle(cornerRadius: 28))
                .padding(.bottom, 36)
        }
        .padding(.horizontal, 24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.ignoresSafeArea())
        .animation(.easeInOut(duration: 0.25), value: phaseText)
    }

    // The three tests: current one highlighted, finished ones ticked.
    private var steps: some View {
        HStack(spacing: 10) {
            ForEach(Self.order, id: \.self) { s in
                let current = s == streamer.bessStance
                let done = streamer.bessCompleted.contains(s)
                HStack(spacing: 6) {
                    if done && !current {
                        Image(systemName: "checkmark.circle.fill")
                    }
                    Text(Self.title(s))
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                }
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(current ? .black : .white.opacity(done ? 0.9 : 0.55))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity)
                .background(current ? Color.white : Color.white.opacity(0.12), in: Capsule())
            }
        }
    }

    private var phaseText: String {
        if streamer.bessJustFinished != nil, streamer.bessPhase == "idle" {
            return "Done. You can relax."
        }
        switch streamer.bessPhase {
        case "countdown":
            return streamer.bessWaitingForView
                ? "Step back so your whole body is in view"
                : "Get ready"
        case "running":
            return "Hold still"
        default:
            return "Wait for the doctor"
        }
    }

    private var phaseColor: Color {
        if streamer.bessJustFinished != nil, streamer.bessPhase == "idle" { return .blue }
        switch streamer.bessPhase {
        case "countdown": return .orange
        case "running": return .green
        default: return .gray
        }
    }

    private static func title(_ stance: String) -> String {
        switch stance {
        case "double": return "Feet together"
        case "tandem": return "Tandem"
        case "single": return "Single leg"
        default: return stance
        }
    }

    /// Plain instructions. Left / right are the patient's own legs.
    private func instructions(_ stance: String) -> String {
        let back = streamer.bessNondominant                     // non-dominant leg
        let front = back == "left" ? "right" : "left"
        switch stance {
        case "double":
            return "Feet together, touching.\nHands on hips. Eyes closed."
        case "tandem":
            return "\(front.capitalized) foot in front, \(back) foot behind, heel to toe.\nHands on hips. Eyes closed."
        case "single":
            return "Stand on your \(back) leg.\nLift your \(front) foot.\nHands on hips. Eyes closed."
        default:
            return "Hands on hips. Eyes closed."
        }
    }
}

// Shown during the call, between stances: the three tests with short
// instructions, ticked when done. The doctor starts each one.

struct BalanceTestsPanel: View {
    @ObservedObject var streamer: VideoStreamClient

    private static let tests: [(id: String, title: String)] = [
        ("double", "Feet together"),
        ("tandem", "Tandem"),
        ("single", "Single leg"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Balance test")
                .font(.system(size: 20, weight: .bold, design: .rounded))
                .foregroundStyle(.white)

            ForEach(Self.tests, id: \.id) { test in
                let done = streamer.bessCompleted.contains(test.id)
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: done ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 20))
                        .foregroundStyle(done ? Color.green : Color.white.opacity(0.6))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(test.title)
                            .font(.system(size: 18, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white)
                        Text(instructions(test.id))
                            .font(.system(size: 15, design: .rounded))
                            .foregroundStyle(.white.opacity(0.8))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            Text("Your doctor will start each test.")
                .font(.system(size: 15, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 18))
    }

    private func instructions(_ id: String) -> String {
        let back = streamer.bessNondominant
        let front = back == "left" ? "right" : "left"
        switch id {
        case "double": return "Feet together, hands on hips, eyes closed."
        case "tandem": return "\(front.capitalized) foot in front, \(back) foot behind, hands on hips, eyes closed."
        default: return "Stand on your \(back) leg, hands on hips, eyes closed."
        }
    }
}
