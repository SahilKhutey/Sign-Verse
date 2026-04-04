import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { DashboardLayout } from "@/components/layout/dashboard-layout"
const inter = Inter({ subsets: ["latin"] })
export const metadata: Metadata = {
  title: "SignVerse - AI Perception Control System",
  description: "Real-time monitoring and control for AI perception systems",
}
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}><DashboardLayout>{children}</DashboardLayout></body>
    </html>
  )
}