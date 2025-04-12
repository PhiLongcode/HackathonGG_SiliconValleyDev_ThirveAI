
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Calendar } from "@/components/ui/calendar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Video, MessageSquare, Calendar as CalendarIcon, Clock, Plus } from "lucide-react";
import { addDays } from "date-fns";

const appointments = [
  { id: 1, patient: "Emma Johnson", time: "10:00 AM", duration: "45 min", type: "Video Call", status: "upcoming" },
  { id: 2, patient: "David Kim", time: "11:30 AM", duration: "30 min", type: "Chat Session", status: "upcoming" },
  { id: 3, patient: "Sophia Rodriguez", time: "2:15 PM", duration: "45 min", type: "Video Call", status: "upcoming" },
  { id: 4, patient: "Michael Chen", time: "4:00 PM", duration: "60 min", type: "Video Call", status: "upcoming" },
];

export const Appointments = () => {
  const today = new Date();
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Appointments</h2>
          <p className="text-muted-foreground">Manage your schedule and patient sessions.</p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          New Appointment
        </Button>
      </div>
      
      <div className="grid gap-6 md:grid-cols-12">
        <Card className="md:col-span-8">
          <CardHeader>
            <Tabs defaultValue="upcoming">
              <TabsList>
                <TabsTrigger value="upcoming">Upcoming</TabsTrigger>
                <TabsTrigger value="past">Past</TabsTrigger>
                <TabsTrigger value="all">All</TabsTrigger>
              </TabsList>
            
              <TabsContent value="upcoming">
                <div className="space-y-4 mt-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {appointments.map((appointment) => (
                      <div 
                        key={appointment.id} 
                        className="flex p-4 border rounded-lg bg-card hover:shadow-sm transition-shadow"
                      >
                        <div className="mr-4 flex-shrink-0">
                          <Avatar className="h-12 w-12">
                            <AvatarFallback>{appointment.patient.charAt(0)}</AvatarFallback>
                          </Avatar>
                        </div>
                        <div className="flex-grow space-y-1">
                          <h4 className="font-medium">{appointment.patient}</h4>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            <span>{appointment.time} ({appointment.duration})</span>
                          </div>
                          <div className="flex items-center justify-between mt-2">
                            <Badge variant="outline" className="flex items-center gap-1">
                              {appointment.type === "Video Call" ? (
                                <Video className="h-3 w-3" />
                              ) : (
                                <MessageSquare className="h-3 w-3" />
                              )}
                              {appointment.type}
                            </Badge>
                            <Button size="sm" variant="outline">
                              {appointment.type === "Video Call" ? "Join" : "Chat"}
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="past">
                <div className="p-4 text-center text-muted-foreground">
                  No past appointments to display.
                </div>
              </TabsContent>
              
              <TabsContent value="all">
                <div className="space-y-4 mt-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {appointments.map((appointment) => (
                      <div 
                        key={appointment.id} 
                        className="flex p-4 border rounded-lg bg-card hover:shadow-sm transition-shadow"
                      >
                        <div className="mr-4 flex-shrink-0">
                          <Avatar className="h-12 w-12">
                            <AvatarFallback>{appointment.patient.charAt(0)}</AvatarFallback>
                          </Avatar>
                        </div>
                        <div className="flex-grow space-y-1">
                          <h4 className="font-medium">{appointment.patient}</h4>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            <span>{appointment.time} ({appointment.duration})</span>
                          </div>
                          <div className="flex items-center justify-between mt-2">
                            <Badge variant="outline" className="flex items-center gap-1">
                              {appointment.type === "Video Call" ? (
                                <Video className="h-3 w-3" />
                              ) : (
                                <MessageSquare className="h-3 w-3" />
                              )}
                              {appointment.type}
                            </Badge>
                            <Button size="sm" variant="outline">
                              {appointment.type === "Video Call" ? "Join" : "Chat"}
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardHeader>
        </Card>
        
        <Card className="md:col-span-4">
          <CardHeader>
            <CardTitle className="flex items-center">
              <CalendarIcon className="mr-2 h-5 w-5 text-muted-foreground" />
              Calendar
            </CardTitle>
            <CardDescription>Your schedule for the week</CardDescription>
          </CardHeader>
          <CardContent>
            <Calendar
              mode="single"
              selected={today}
              className="rounded-md border"
            />
            <div className="mt-4 space-y-2">
              <h4 className="font-medium text-sm">Today's Schedule</h4>
              {appointments.slice(0, 3).map((appointment, index) => (
                <div key={index} className="flex items-center justify-between text-sm p-2 rounded-md hover:bg-muted/50">
                  <span>{appointment.time}</span>
                  <span className="text-muted-foreground truncate max-w-[100px]">
                    {appointment.patient}
                  </span>
                </div>
              ))}
            </div>
            <Button variant="outline" className="w-full mt-4">View Full Schedule</Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
