
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Video, Mic, MicOff, Camera, CameraOff, Phone, MessageSquare, FileText, Users } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export const VideoConsultation = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Video Consultation</h2>
        <p className="text-muted-foreground">Connect with your patients through secure video calls.</p>
      </div>
      
      <div className="grid gap-6 grid-cols-1 2xl:grid-cols-4">
        <Card className="2xl:col-span-3">
          <CardContent className="p-0 overflow-hidden">
            <div className="bg-zinc-900 aspect-video flex items-center justify-center relative">
              <div className="text-white">
                <p className="text-center text-lg">Camera placeholder</p>
                <p className="text-center text-muted-foreground">
                  Your video call will appear here when connected
                </p>
              </div>
              
              {/* Small camera view */}
              <div className="absolute bottom-4 right-4 w-40 h-28 bg-zinc-800 rounded-lg flex items-center justify-center">
                <p className="text-white text-xs">Your camera</p>
              </div>
              
              {/* Call controls */}
              <div className="absolute bottom-4 left-0 right-0 mx-auto w-fit flex gap-2 p-2 bg-zinc-800/60 backdrop-blur-sm rounded-full">
                <Button size="icon" variant="ghost" className="rounded-full bg-zinc-700 hover:bg-zinc-600 text-white">
                  <Mic className="h-5 w-5" />
                </Button>
                <Button size="icon" variant="ghost" className="rounded-full bg-zinc-700 hover:bg-zinc-600 text-white">
                  <Video className="h-5 w-5" />
                </Button>
                <Button size="icon" variant="destructive" className="rounded-full">
                  <Phone className="h-5 w-5" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="2xl:col-span-1">
          <CardHeader className="p-4">
            <Tabs defaultValue="chat">
              <TabsList className="grid grid-cols-3">
                <TabsTrigger value="chat"><MessageSquare className="h-4 w-4" /></TabsTrigger>
                <TabsTrigger value="notes"><FileText className="h-4 w-4" /></TabsTrigger>
                <TabsTrigger value="info"><Users className="h-4 w-4" /></TabsTrigger>
              </TabsList>
            
              <TabsContent value="chat" className="m-0">
                <div className="h-[calc(60vh-180px)] overflow-y-auto flex flex-col gap-3 mb-4">
                  <div className="flex gap-2">
                    <Avatar className="h-8 w-8 mt-1">
                      <AvatarFallback>P</AvatarFallback>
                    </Avatar>
                    <div className="bg-muted p-2 rounded-lg max-w-[80%]">
                      <p className="text-sm">Hello doctor, I've been feeling much better since our last session.</p>
                    </div>
                  </div>
                  
                  <div className="flex gap-2 flex-row-reverse">
                    <Avatar className="h-8 w-8 mt-1">
                      <AvatarFallback>DR</AvatarFallback>
                    </Avatar>
                    <div className="bg-primary/10 p-2 rounded-lg max-w-[80%]">
                      <p className="text-sm">That's great to hear! Have you been practicing the exercises we discussed?</p>
                    </div>
                  </div>
                  
                  <div className="flex gap-2">
                    <Avatar className="h-8 w-8 mt-1">
                      <AvatarFallback>P</AvatarFallback>
                    </Avatar>
                    <div className="bg-muted p-2 rounded-lg max-w-[80%]">
                      <p className="text-sm">Yes, I've been doing the breathing exercises every morning and it's helped a lot with my anxiety.</p>
                    </div>
                  </div>
                </div>
                
                <div className="flex gap-2">
                  <Input
                    placeholder="Type a message..."
                    className="flex-1"
                  />
                  <Button size="icon">
                    <MessageSquare className="h-4 w-4" />
                  </Button>
                </div>
              </TabsContent>
              
              <TabsContent value="notes" className="m-0">
                <div className="space-y-4">
                  <h4 className="font-medium">Session Notes</h4>
                  <textarea
                    className="w-full h-[calc(60vh-200px)] p-3 border rounded-md"
                    placeholder="Take notes during your session here..."
                  />
                  <Button className="w-full">Save Notes</Button>
                </div>
              </TabsContent>
              
              <TabsContent value="info" className="m-0">
                <div className="space-y-4">
                  <div className="flex flex-col items-center gap-2 p-4">
                    <Avatar className="h-16 w-16">
                      <AvatarFallback>EM</AvatarFallback>
                    </Avatar>
                    <h3 className="font-medium">Emma Johnson</h3>
                    <p className="text-sm text-muted-foreground">28 years old</p>
                    <Badge>Anxiety</Badge>
                  </div>
                  
                  <div className="border-t pt-4">
                    <h4 className="font-medium mb-2">Patient Info</h4>
                    <dl className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Last Session:</dt>
                        <dd>April 3, 2025</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Status:</dt>
                        <dd>Needs Attention</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Risk Level:</dt>
                        <dd>Medium</dd>
                      </div>
                    </dl>
                  </div>
                  
                  <Button variant="outline" className="w-full">View Full Profile</Button>
                </div>
              </TabsContent>
            </Tabs>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            {/* Empty content since we moved the tabs content inside the Tabs component */}
          </CardContent>
        </Card>
      </div>
      
      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Upcoming Calls</CardTitle>
            <CardDescription>
              Your scheduled video consultations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <Avatar className="h-10 w-10">
                    <AvatarFallback>MC</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-medium">Michael Chen</p>
                    <p className="text-xs text-muted-foreground">Today, 4:00 PM (60 min)</p>
                  </div>
                </div>
                <Button size="sm">Join</Button>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <Avatar className="h-10 w-10">
                    <AvatarFallback>SR</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-medium">Sophia Rodriguez</p>
                    <p className="text-xs text-muted-foreground">Tomorrow, 2:15 PM (45 min)</p>
                  </div>
                </div>
                <Button size="sm" variant="outline">Schedule</Button>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Call Settings</CardTitle>
            <CardDescription>
              Configure your video consultation preferences
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg flex flex-col items-center gap-3">
                  <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center">
                    <Camera className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-medium">Camera Settings</h3>
                  <Button variant="outline" size="sm">Configure</Button>
                </div>
                
                <div className="p-4 border rounded-lg flex flex-col items-center gap-3">
                  <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center">
                    <Mic className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-medium">Audio Settings</h3>
                  <Button variant="outline" size="sm">Configure</Button>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-muted-foreground" />
                  <span>Auto-save session notes</span>
                </div>
                <Button size="sm" variant="ghost">
                  Configure
                </Button>
              </div>
              
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-2">
                  <Video className="h-5 w-5 text-muted-foreground" />
                  <span>Use HD video when available</span>
                </div>
                <Button size="sm" variant="ghost">
                  Configure
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
