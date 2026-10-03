//
//  LandmarkOverlay.swift
//  NoseThumb
//

import SwiftUI

// Draws dots for tracked body points on top of the camera preview.
struct LandmarkOverlay: View {
    let pose: PoseDetector

    var body: some View {
        // Read the values here, outside GeometryReader, so SwiftUI
        // reliably redraws this view whenever they change.
        let bridgeFront = pose.bridgeFront
        let leftShoulder = pose.leftShoulder
        let rightShoulder = pose.rightShoulder
        let thumb = pose.thumb
        let imageSize = pose.imageSize

        GeometryReader { geometry in
            let viewSize = geometry.size
            ZStack {
                dot(bridgeFront, color: .green, imageSize: imageSize, viewSize: viewSize)
                dot(leftShoulder, color: .cyan, imageSize: imageSize, viewSize: viewSize)
                dot(rightShoulder, color: .cyan, imageSize: imageSize, viewSize: viewSize)
                dot(thumb, color: .pink, imageSize: imageSize, viewSize: viewSize)
            }
        }
    }

    @ViewBuilder
    private func dot(_ point: TrackedPoint?, color: Color, imageSize: CGSize, viewSize: CGSize) -> some View {
        if let point, point.visibility >= PoseDetector.minVisibility, imageSize != .zero {
            let position = screenPosition(of: point, imageSize: imageSize, viewSize: viewSize)
            ZStack {
                Circle()
                    .fill(color)
                    .frame(width: 16, height: 16)
                    .overlay(Circle().stroke(.black, lineWidth: 2))
                // Temporary: show the visibility score so we can check its range.
                Text(String(format: "%.2f", point.visibility))
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .shadow(color: .black, radius: 2)
                    .offset(y: -18)
            }
            .position(position)
        }
    }

    // Converts a 0...1 image point to screen coordinates, using the same
    // "aspect fill" scaling the camera preview uses (scale up, crop edges).
    private func screenPosition(of point: TrackedPoint, imageSize: CGSize, viewSize: CGSize) -> CGPoint {
        let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        let offsetX = (viewSize.width - scaledWidth) / 2
        let offsetY = (viewSize.height - scaledHeight) / 2
        return CGPoint(x: point.x * scaledWidth + offsetX,
                       y: point.y * scaledHeight + offsetY)
    }
}
