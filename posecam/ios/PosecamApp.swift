// PosecamApp.swift
// App entry point: the Modified BESS balance test, with one server connection
// and the LiDAR camera.
//
// Setup (Xcode 26, iOS 17+): see "iPhone app" in posecam/README.md.

import SwiftUI

@main
struct PosecamApp: App {
    @StateObject private var client = PosecamClient()
    @StateObject private var camera = DepthCamera()

    var body: some Scene {
        WindowGroup {
            BessTestView(client: client, camera: camera)
                .onAppear {
                let client = client
                // camera queue -> build the JSON there -> send on the main actor
                camera.setFrameHandler { frame in
                    guard let text = PosecamClient.frameMessage(frame) else { return }
                    Task { @MainActor in client.sendFrameMessage(text) }
                }
            }
        }
    }
}
