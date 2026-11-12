// Build with: swiftc figures/render_pdf.swift -o /tmp/render_clinical_pdf
// Usage: /tmp/render_clinical_pdf input.pdf output.png dpi
import AppKit
import Foundation
import CoreGraphics

let input = CommandLine.arguments[1]
let output = CommandLine.arguments[2]
let dpi = Double(CommandLine.arguments[3])!
guard let doc = CGPDFDocument(URL(fileURLWithPath: input) as CFURL),
      let page = doc.page(at: 1) else { fatalError("Cannot read PDF") }
let bounds = page.getBoxRect(.mediaBox)
let scale = dpi / 72.0
let width = Int(ceil(bounds.width * scale))
let height = Int(ceil(bounds.height * scale))
guard let ctx = CGContext(data: nil, width: width, height: height,
                          bitsPerComponent: 8, bytesPerRow: width * 4,
                          space: CGColorSpaceCreateDeviceRGB(),
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    fatalError("Cannot create drawing context")
}
ctx.setFillColor(NSColor.white.cgColor)
ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
ctx.scaleBy(x: scale, y: scale)
ctx.drawPDFPage(page)
guard let image = ctx.makeImage() else { fatalError("Cannot render PDF") }
let rep = NSBitmapImageRep(cgImage: image)
rep.size = NSSize(width: bounds.width, height: bounds.height)
guard let data = rep.representation(using: .png, properties: [:]) else {
    fatalError("Cannot encode PNG")
}
try data.write(to: URL(fileURLWithPath: output))
print("Rendered \(width)x\(height): \(output)")
