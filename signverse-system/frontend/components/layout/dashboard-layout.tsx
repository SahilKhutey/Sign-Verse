"use client"
import { useState } from "react"
import { Sidebar } from "./sidebar"
import { Header } from "./header"
import { useSystemStore } from "@/store/use-system-store"
interface DashboardLayoutProps { children: React.ReactNode }
export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [open, setOpen] = useState(false)
  const { darkMode } = useSystemStore()
  return (
    <div className={`min-h-screen ${darkMode ? "dark" : ""}`}>
      <Sidebar open={open} onClose={() => setOpen(false)} />
      <div className="lg:pl-72">
        <Header onMenuClick={() => setOpen(true)} />
        <main className="py-8"><div className="px-4 sm:px-6 lg:px-8">{children}</div></main>
      </div>
    </div>
  )
}