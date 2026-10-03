//
//  ContentView.swift
//  NoseThumb
//
//  Created by Aishwarya patil on 10/3/26.
//

import SwiftUI

struct ContentView: View {
    @State private var camera = CameraManager()
    // Patient view: camera + one instruction, no data. Clinician view: dots, numbers, debug.
    @State private var patientMode = false

    var body: some View {
        cameraScreen
    }

    @ViewBuilder
    private var cameraScreen: some View {
        let pose = camera.poseDetector

        // Preview and dots share the exact same full-screen area, so positions line up.
        ZStack {
            CameraPreview(session: camera.session)
            if !patientMode {
                LandmarkOverlay(pose: pose)
            }
        }
            .ignoresSafeArea()
            .overlay(alignment: .top) {
                if patientMode {
                    // Patient view: one big instruction, plus whether the clinician is on the call.
                    VStack(spacing: 8) {
                        Text(patientInstruction(pose))
                            .font(.system(size: 28, weight: .bold, design: .rounded))
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.white)
                            .padding(20)
                            .frame(maxWidth: .infinity)
                            .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 18))
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                } else {
                    clinicianPanel(pose)
                }
            }
            .overlay(alignment: .bottom) {
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        overlayButton(camera.position == .front ? "Back camera" : "Front camera",
                                      systemImage: "arrow.triangle.2.circlepath.camera") {
                            camera.switchCamera()
                        }
                        overlayButton(patientMode ? "Clinician view" : "Patient view",
                                      systemImage: patientMode ? "stethoscope" : "person") {
                            patientMode.toggle()
                        }
                    }

                    if !patientMode {
                        // Temporary debug info.
                        Text(camera.debugText)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(.white)
                            .padding(8)
                            .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 8))
                    }
                }
                .padding(.bottom, 24)
            }
            .onAppear {
                camera.start()
            }
    }

    // All tracking values in one dictionary, e.g. to send to a dashboard. Missing values are
    // NSNull, so it can be turned into JSON directly (JSONSerialization). Not used by this
    // screen; it's an example of what to read from PoseDetector and CameraManager.
    func trackingSnapshot(_ pose: PoseDetector) -> [String: Any] {
        let faceVisible = pose.personDetected && (pose.nose?.visibility ?? 0) >= PoseDetector.minVisibility
        return [
            "distanceCM": pose.bridgeThumbCM ?? NSNull(),
            "bridgeDepthCM": pose.bridgeDepthCM ?? NSNull(),
            "faceVisible": faceVisible,
            "isProfile": pose.isProfile,
            "thumbVisible": pose.thumb != nil,
            "camera": camera.position == .back ? "back" : "front",
            "instruction": patientInstruction(pose),
            // Dot positions for the doctor's video overlay (0...1 of the video frame).
            "bridge": pose.bridgeFront.map { ["x": $0.x, "y": $0.y] } ?? NSNull(),
            "thumb": pose.thumb.map { ["x": $0.x, "y": $0.y] } ?? NSNull(),
        ]
    }

    // Clinician view: status, debug and the measurement.
    private func clinicianPanel(_ pose: PoseDetector) -> some View {
        VStack(spacing: 4) {
            // Small status line: FPS, person, depth accuracy.
            HStack(spacing: 12) {
                Text("\(camera.fps) FPS (depth \(camera.depthFPS), valid \(camera.validDepthPercent)%)")
                    .monospacedDigit()
                if pose.isLoaded {
                    Text(pose.personDetected ? "Person ✓" : "No person")
                        .foregroundStyle(pose.personDetected ? .green : .orange)
                    Text("depth: \(pose.depthAccuracy)")
                } else {
                    Text("Pose model failed to load")
                        .foregroundStyle(.red)
                }
                if !pose.handLoaded {
                    Text("Hand model failed to load")
                        .foregroundStyle(.red)
                }
            }
            .font(.system(size: 16, weight: .semibold, design: .rounded))

            // Temporary debug: how the bridge was found, or why not.
            Text(pose.bridgeStatus)
                .font(.system(size: 13, weight: .medium, design: .monospaced))

            if pose.isLoaded {
                // The two depths the gap is computed from.
                HStack(spacing: 16) {
                    Text(bridgeDepthText(pose))
                    Text(thumbDepthText(pose))
                }
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .monospacedDigit()

                // The measurement: the biggest text on screen.
                Text(bridgeThumbText(pose))
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .monospacedDigit()
            }
        }
        .lineLimit(1)
        .minimumScaleFactor(0.6)
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
        .padding(.top, 8)
    }

    // A small dark pill button used on top of the camera image.
    private func overlayButton(_ title: String, systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 16, weight: .semibold, design: .rounded))
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.black.opacity(0.6), in: Capsule())
                .foregroundStyle(.white)
        }
    }

    // What the patient should do next, based on what the camera sees. Checked in order:
    // the first thing that isn't right yet is the instruction.
    private func patientInstruction(_ pose: PoseDetector) -> String {
        guard pose.personDetected, let nose = pose.nose, nose.visibility >= PoseDetector.minVisibility else {
            return "Move so your face is in view"
        }
        guard pose.isProfile else {
            return "Turn so the side of your face points at the phone"
        }
        guard pose.thumb != nil else {
            return "Hold your thumb up at arm's length, level with your eyes"
        }
        guard pose.bridgeThumbCM != nil else {
            return "Hold still for a moment"
        }
        return "Slowly bring your thumb toward your nose. Say \u{201C}now\u{201D} when you see double."
    }

    // Each text shows a clear message instead of a wrong number.

    // "Bridge 54.1 cm": distance from the phone to the front of the nose bridge.
    private func bridgeDepthText(_ pose: PoseDetector) -> String {
        guard pose.bridgeFront != nil else {
            return "Bridge: not found"
        }
        guard let cm = pose.bridgeDepthCM else {
            return "Bridge: no depth"
        }
        return String(format: "Bridge %.1f cm", cm)
    }

    // "Thumb 30.1 cm"
    private func thumbDepthText(_ pose: PoseDetector) -> String {
        guard pose.thumb != nil else {
            return "Thumb: not visible"
        }
        guard let cm = pose.thumbDepthCM else {
            return "Thumb: no depth"
        }
        return String(format: "Thumb %.1f cm", cm)
    }

    // "Bridge–thumb 12.3 cm": measured from the bridge of the nose. Only when everything needed is valid.
    private func bridgeThumbText(_ pose: PoseDetector) -> String {
        guard let cm = pose.bridgeThumbCM else {
            return "Bridge–thumb: –"
        }
        return String(format: "Bridge–thumb %.1f cm", cm)
    }
}
