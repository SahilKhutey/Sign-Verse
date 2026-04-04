'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { Users, Activity, Eye, Clock, TrendingUp, BarChart3 } from 'lucide-react'

const mockTimeSeriesData = [
  { time: '00:00', people: 12, objects: 8, activity: 24 },
  { time: '01:00', people: 8, objects: 6, activity: 18 },
  { time: '02:00', people: 4, objects: 3, activity: 9 },
  { time: '03:00', people: 2, objects: 2, activity: 5 },
  { time: '04:00', people: 3, objects: 1, activity: 4 },
  { time: '05:00', people: 6, objects: 4, activity: 12 },
  { time: '06:00', people: 15, objects: 10, activity: 30 },
  { time: '07:00', people: 28, objects: 15, activity: 56 },
  { time: '08:00', people: 35, objects: 20, activity: 70 },
  { time: '09:00', people: 32, objects: 18, activity: 64 },
  { time: '10:00', people: 30, objects: 16, activity: 60 },
]

const activityDistribution = [
  { name: 'Walking', value: 45 },
  { name: 'Standing', value: 25 },
  { name: 'Sitting', value: 15 },
  { name: 'Interacting', value: 10 },
  { name: 'Running', value: 5 },
]

const COLORS = ['#6366F1', '#22C55E', '#EF4444', '#F59E0B', '#8B5CF6']

export function MetricsDashboard() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Advanced Analytics</h2>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-400">
              Active Persons
            </CardTitle>
            <Users className="h-4 w-4 text-signverse-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">24</div>
            <p className="text-xs text-slate-400">+12% from yesterday</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-400">
              Object Count
            </CardTitle>
            <Eye className="h-4 w-4 text-signverse-accent" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">16</div>
            <p className="text-xs text-slate-400">+5% from yesterday</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-400">
              Avg. Velocity
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">1.2m/s</div>
            <p className="text-xs text-slate-400">-0.3m/s from yesterday</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-400">
              Peak Activity
            </CardTitle>
            <Activity className="h-4 w-4 text-orange-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">08:00</div>
            <p className="text-xs text-slate-400">70 activities detected</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Activity Timeline */}
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white">Activity Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockTimeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    borderColor: '#374151',
                    color: '#F9FAFB'
                  }} 
                />
                <Line type="monotone" dataKey="people" stroke="#6366F1" strokeWidth={2} />
                <Line type="monotone" dataKey="objects" stroke="#22C55E" strokeWidth={2} />
                <Line type="monotone" dataKey="activity" stroke="#F59E0B" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Activity Distribution */}
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white">Activity Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={activityDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {activityDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    borderColor: '#374151',
                    color: '#F9FAFB'
                  }} 
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Motion Heatmap */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Motion Heatmap</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <div className="text-white text-center">
              <BarChart3 className="h-12 w-12 mx-auto mb-2" />
              <p>Motion heatmap visualization</p>
              <p className="text-sm text-slate-200">(Would show spatial activity distribution)</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
