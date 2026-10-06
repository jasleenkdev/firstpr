import type { Metadata, Viewport } from "next";
import { Schibsted_Grotesk } from "next/font/google";
import "./globals.css";

const sans = Schibsted_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "FirstPR: find your first contribution",
  description:
    "Open-source projects and beginner issues picked for your skills and your time, with the reason behind each pick.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#F6F8F9" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={sans.variable}>
      <body>{children}</body>
    </html>
  );
}
