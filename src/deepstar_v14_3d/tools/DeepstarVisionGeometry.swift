import CoreImage
import CoreVideo
import Foundation
import Vision

struct FaceBox: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
    let confidence: Double
}

struct Analysis: Codable {
    let engine: String
    let maskWidth: Int
    let maskHeight: Int
    let subjectPixelCount: Int
    let faces: [FaceBox]
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

guard CommandLine.arguments.count == 4 else {
    fail("usage: DeepstarVisionGeometry image mask.pgm analysis.json")
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
let maskURL = URL(fileURLWithPath: CommandLine.arguments[2])
let analysisURL = URL(fileURLWithPath: CommandLine.arguments[3])

guard let image = CIImage(contentsOf: imageURL) else {
    fail("Apple Vision could not decode the image")
}

let segmentation = VNGeneratePersonSegmentationRequest()
segmentation.qualityLevel = .accurate
segmentation.outputPixelFormat = kCVPixelFormatType_OneComponent8
let faces = VNDetectFaceRectanglesRequest()
let handler = VNImageRequestHandler(ciImage: image, options: [:])

do {
    try handler.perform([segmentation, faces])
} catch {
    fail("Apple Vision analysis failed: \(error.localizedDescription)")
}

guard let observation = segmentation.results?.first else {
    fail("Apple Vision returned no person mask")
}

let buffer = observation.pixelBuffer
CVPixelBufferLockBaseAddress(buffer, .readOnly)
defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

guard let base = CVPixelBufferGetBaseAddress(buffer) else {
    fail("Apple Vision returned an unreadable person mask")
}

let width = CVPixelBufferGetWidth(buffer)
let height = CVPixelBufferGetHeight(buffer)
let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
let source = base.assumingMemoryBound(to: UInt8.self)
var mask = Data(capacity: width * height)
var subjectPixelCount = 0

for y in 0..<height {
    let row = source.advanced(by: y * bytesPerRow)
    for x in 0..<width {
        let value = row[x]
        mask.append(value)
        if value >= 32 { subjectPixelCount += 1 }
    }
}

var pgm = Data("P5\n\(width) \(height)\n255\n".utf8)
pgm.append(mask)

do {
    try pgm.write(to: maskURL, options: .atomic)
} catch {
    fail("Could not write Apple Vision mask: \(error.localizedDescription)")
}

let faceBoxes = (faces.results ?? []).map { observation in
    let box = observation.boundingBox
    return FaceBox(
        x: box.origin.x,
        y: box.origin.y,
        width: box.width,
        height: box.height,
        confidence: Double(observation.confidence)
    )
}

let result = Analysis(
    engine: "apple-vision-person-segmentation-v1",
    maskWidth: width,
    maskHeight: height,
    subjectPixelCount: subjectPixelCount,
    faces: faceBoxes
)

do {
    let encoded = try JSONEncoder().encode(result)
    try encoded.write(to: analysisURL, options: .atomic)
} catch {
    fail("Could not write Apple Vision analysis: \(error.localizedDescription)")
}
