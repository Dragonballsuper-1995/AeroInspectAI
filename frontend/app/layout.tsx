import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AeroInspect AI | Infrastructure Inspection",
  description: "Edge computer vision console for autonomous infrastructure inspection.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}