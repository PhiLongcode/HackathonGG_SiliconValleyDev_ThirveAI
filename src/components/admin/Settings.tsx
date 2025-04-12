
import React from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export const Settings = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
        <p className="text-muted-foreground">Manage your system settings and preferences.</p>
      </div>
      
      <Tabs defaultValue="general" className="space-y-4">
        <TabsList>
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="ai">AI Settings</TabsTrigger>
        </TabsList>
        
        <TabsContent value="general" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
              <CardDescription>
                Manage your platform-wide settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="platform-name">Platform Name</Label>
                <Input id="platform-name" defaultValue="ThriveAI" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="contact-email">Contact Email</Label>
                <Input id="contact-email" defaultValue="admin@thriveai.com" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="language">Default Language</Label>
                <Select defaultValue="en">
                  <SelectTrigger id="language">
                    <SelectValue placeholder="Select language" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="vi">Vietnamese</SelectItem>
                    <SelectItem value="fr">French</SelectItem>
                    <SelectItem value="de">German</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="maintenance-mode">Maintenance Mode</Label>
                  <p className="text-xs text-muted-foreground">
                    Temporarily disable user access to the platform
                  </p>
                </div>
                <Switch id="maintenance-mode" />
              </div>
            </CardContent>
            <CardFooter>
              <Button>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        <TabsContent value="security" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Security Settings</CardTitle>
              <CardDescription>
                Manage your security preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="two-factor">Two-Factor Authentication</Label>
                  <p className="text-xs text-muted-foreground">
                    Require an additional authentication step for all admin accounts
                  </p>
                </div>
                <Switch id="two-factor" defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="session-timeout">Auto Logout</Label>
                  <p className="text-xs text-muted-foreground">
                    Automatically log out inactive admin sessions after 30 minutes
                  </p>
                </div>
                <Switch id="session-timeout" defaultChecked />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password-policy">Password Policy</Label>
                <Select defaultValue="strong">
                  <SelectTrigger id="password-policy">
                    <SelectValue placeholder="Select policy" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="basic">Basic (8+ characters)</SelectItem>
                    <SelectItem value="medium">Medium (8+ chars, 1 number)</SelectItem>
                    <SelectItem value="strong">Strong (8+ chars, uppercase, number, symbol)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
            <CardFooter>
              <Button>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        <TabsContent value="notifications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Notification Settings</CardTitle>
              <CardDescription>
                Configure how and when you receive notifications
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="email-notifications">Email Notifications</Label>
                  <p className="text-xs text-muted-foreground">
                    Receive system alerts and reports via email
                  </p>
                </div>
                <Switch id="email-notifications" defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="user-registrations">New User Registrations</Label>
                  <p className="text-xs text-muted-foreground">
                    Get notified when new users register
                  </p>
                </div>
                <Switch id="user-registrations" />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="system-alerts">System Alerts</Label>
                  <p className="text-xs text-muted-foreground">
                    Receive critical system alerts and warnings
                  </p>
                </div>
                <Switch id="system-alerts" defaultChecked />
              </div>
            </CardContent>
            <CardFooter>
              <Button>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        <TabsContent value="ai" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Settings</CardTitle>
              <CardDescription>
                Configure how the AI interacts with users
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="ai-suggestions">AI Suggestions</Label>
                  <p className="text-xs text-muted-foreground">
                    Allow AI to proactively suggest mental health exercises to users
                  </p>
                </div>
                <Switch id="ai-suggestions" defaultChecked />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ai-model">AI Model</Label>
                <Select defaultValue="advanced">
                  <SelectTrigger id="ai-model">
                    <SelectValue placeholder="Select AI model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="basic">Basic (Faster responses)</SelectItem>
                    <SelectItem value="balanced">Balanced</SelectItem>
                    <SelectItem value="advanced">Advanced (Most accurate)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="data-collection">Data Collection</Label>
                  <p className="text-xs text-muted-foreground">
                    Collect anonymized data to improve AI performance
                  </p>
                </div>
                <Switch id="data-collection" defaultChecked />
              </div>
            </CardContent>
            <CardFooter>
              <Button>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
