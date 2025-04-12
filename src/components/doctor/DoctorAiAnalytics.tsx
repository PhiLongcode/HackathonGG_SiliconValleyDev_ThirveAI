
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer } from "@/components/ui/chart";
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const treatmentData = [
  { name: 'CBT', value: 45 },
  { name: 'Mindfulness', value: 25 },
  { name: 'Medication', value: 15 },
  { name: 'Other', value: 15 },
];

const patientImprovementData = [
  { month: 'Jan', anxiety: 75, depression: 85, baseline: 100 },
  { month: 'Feb', anxiety: 70, depression: 80, baseline: 100 },
  { month: 'Mar', anxiety: 65, depression: 70, baseline: 100 },
  { month: 'Apr', anxiety: 60, depression: 65, baseline: 100 },
  { month: 'May', anxiety: 55, depression: 55, baseline: 100 },
  { month: 'Jun', anxiety: 45, depression: 50, baseline: 100 },
];

const COLORS = ['#9F7AEA', '#68D391', '#4299E1', '#F6AD55'];

export const DoctorAiAnalytics = () => {
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">AI Analytics</h2>
          <p className="text-muted-foreground">AI-powered insights about your patients.</p>
        </div>
        <Select defaultValue="all">
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by condition" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Conditions</SelectItem>
            <SelectItem value="anxiety">Anxiety</SelectItem>
            <SelectItem value="depression">Depression</SelectItem>
            <SelectItem value="stress">Stress</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      <div className="grid gap-6 md:grid-cols-3">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Patient Improvement Trends</CardTitle>
            <CardDescription>
              Symptom reduction over time (lower is better)
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ChartContainer
              config={{
                anxiety: { color: "#9F7AEA" },
                depression: { color: "#4299E1" },
                baseline: { color: "#E5E7EB" }
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={patientImprovementData}
                  margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="month" tick={{ fill: 'var(--muted-foreground)' }} />
                  <YAxis tick={{ fill: 'var(--muted-foreground)' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="baseline" name="Baseline" stroke="#E5E7EB" strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="anxiety" name="Anxiety Symptoms" stroke="#9F7AEA" strokeWidth={2} />
                  <Line type="monotone" dataKey="depression" name="Depression Symptoms" stroke="#4299E1" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Treatment Efficacy</CardTitle>
            <CardDescription>
              Most effective treatments based on AI analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={treatmentData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {treatmentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>AI Insights & Recommendations</CardTitle>
          <CardDescription>
            Personalized treatment suggestions based on patient data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="high-risk">
            <TabsList className="mb-4">
              <TabsTrigger value="high-risk">High Risk Patients</TabsTrigger>
              <TabsTrigger value="recommendations">Treatment Recommendations</TabsTrigger>
              <TabsTrigger value="patterns">Behavioral Patterns</TabsTrigger>
            </TabsList>
            
            <TabsContent value="high-risk" className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <Avatar>
                        <AvatarFallback>JW</AvatarFallback>
                      </Avatar>
                      <div>
                        <h4 className="font-medium">James Wilson</h4>
                        <p className="text-sm text-muted-foreground">PTSD, Severe anxiety</p>
                      </div>
                      <Badge variant="destructive" className="ml-auto">High Risk</Badge>
                    </div>
                    <div className="mt-4">
                      <p className="text-sm">
                        <strong>AI Analysis:</strong> Recent messages show increased distress and sleep disruption. 
                        Language patterns indicate 85% similarity to crisis situations.
                      </p>
                      <p className="text-sm mt-2">
                        <strong>Recommendation:</strong> Immediate check-in, consider crisis intervention protocols.
                      </p>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <Avatar>
                        <AvatarFallback>WK</AvatarFallback>
                      </Avatar>
                      <div>
                        <h4 className="font-medium">William Kim</h4>
                        <p className="text-sm text-muted-foreground">Major depression, Suicidal ideation</p>
                      </div>
                      <Badge variant="destructive" className="ml-auto">High Risk</Badge>
                    </div>
                    <div className="mt-4">
                      <p className="text-sm">
                        <strong>AI Analysis:</strong> Mood tracking shows sharp decline over 72 hours. Journal entries 
                        contain concerning language patterns.
                      </p>
                      <p className="text-sm mt-2">
                        <strong>Recommendation:</strong> Expedite next appointment, evaluate medication efficacy.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            <TabsContent value="recommendations">
              <p className="text-sm text-muted-foreground mb-4">
                AI-generated treatment suggestions based on patient response patterns and clinical research.
              </p>
              
              <ul className="space-y-4">
                <li className="p-4 border rounded-lg">
                  <h4 className="font-medium">Anxiety Disorders</h4>
                  <p className="text-sm mt-1">
                    Patients showing highest improvement with combined CBT and mindfulness approach. 
                    AI suggests structured 8-week program with weekly digital check-ins.
                  </p>
                </li>
                <li className="p-4 border rounded-lg">
                  <h4 className="font-medium">Depression</h4>
                  <p className="text-sm mt-1">
                    Text analysis shows better outcomes when behavioral activation precedes cognitive restructuring. 
                    Recommend starting with activity scheduling before thought challenges.
                  </p>
                </li>
                <li className="p-4 border rounded-lg">
                  <h4 className="font-medium">Stress Management</h4>
                  <p className="text-sm mt-1">
                    Morning breathing exercises correlate with 40% better daily outcomes than evening sessions. 
                    Consider shifting intervention timing to early day for stress patients.
                  </p>
                </li>
              </ul>
            </TabsContent>
            
            <TabsContent value="patterns">
              <p className="text-sm text-muted-foreground mb-4">
                Behavioral patterns detected across your patient population.
              </p>
              
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium">Communication Patterns</h4>
                  <p className="text-sm mt-1">
                    Patients who message within 48 hours of feeling symptoms show 65% faster recovery rates. 
                    Consider encouraging proactive communication protocols.
                  </p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium">Treatment Adherence</h4>
                  <p className="text-sm mt-1">
                    Exercise recommendations have 35% higher adherence when presented with visual aids and progress tracking. 
                    Consider including visual components in homework assignments.
                  </p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium">Relapse Indicators</h4>
                  <p className="text-sm mt-1">
                    Changes in sleep patterns frequently precede mood deterioration by 5-7 days. 
                    Monitor sleep logs closely as early intervention opportunity.
                  </p>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
