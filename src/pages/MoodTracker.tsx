
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Calendar } from "@/components/ui/calendar";
import { Smile, Meh, Frown, Calendar as CalendarIcon } from "lucide-react";

// Mock data for the chart
const mockData = [
  { day: "T2", mood: 7 },
  { day: "T3", mood: 8 },
  { day: "T4", mood: 6 },
  { day: "T5", mood: 5 },
  { day: "T6", mood: 6 },
  { day: "T7", mood: 8 },
  { day: "CN", mood: 9 },
];

const MoodTracker = () => {
  const [date, setDate] = useState<Date | undefined>(new Date());
  const [selectedMood, setSelectedMood] = useState<number | null>(null);
  
  const moods = [
    { value: 2, label: "Rất tệ", icon: <Frown className="h-6 w-6 text-red-500" /> },
    { value: 4, label: "Không tốt", icon: <Frown className="h-6 w-6 text-orange-500" /> },
    { value: 6, label: "Bình thường", icon: <Meh className="h-6 w-6 text-yellow-500" /> },
    { value: 8, label: "Tốt", icon: <Smile className="h-6 w-6 text-lime-500" /> },
    { value: 10, label: "Tuyệt vời", icon: <Smile className="h-6 w-6 text-green-500" /> },
  ];
  
  const getAverageMood = () => {
    const sum = mockData.reduce((acc, curr) => acc + curr.mood, 0);
    return (sum / mockData.length).toFixed(1);
  };
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Biểu đồ tâm trạng</h1>
          <p className="text-muted-foreground">
            Theo dõi cảm xúc hàng ngày và nhận những gợi ý phù hợp
          </p>
        </div>
        <Button className="thrive-button">Đánh giá hôm nay</Button>
      </div>
      
      <Tabs defaultValue="chart" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="chart">Biểu đồ</TabsTrigger>
          <TabsTrigger value="calendar">Lịch</TabsTrigger>
          <TabsTrigger value="stats">Thống kê</TabsTrigger>
        </TabsList>
        
        <TabsContent value="chart" className="p-0 pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Tâm trạng theo tuần</CardTitle>
              <CardDescription>
                Điểm trung bình: {getAverageMood()} / 10
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={mockData}
                    margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis domain={[0, 10]} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="mood"
                      stroke="hsl(var(--primary))"
                      strokeWidth={3}
                      activeDot={{ r: 8 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-6">
                <h4 className="font-medium mb-3">Phân tích</h4>
                <p className="text-muted-foreground">
                  Tâm trạng của bạn có xu hướng tốt hơn vào cuối tuần. Bạn có thể xem xét tìm thêm hoạt động thư giãn vào đầu tuần.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="calendar" className="p-0 pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Lịch tâm trạng</CardTitle>
              <CardDescription>
                Nhấn vào một ngày để thêm hoặc xem tâm trạng
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center p-2">
                <Calendar
                  mode="single"
                  selected={date}
                  onSelect={setDate}
                  className="rounded-md border"
                />
              </div>
              
              {date && (
                <div className="mt-6">
                  <h4 className="font-medium mb-3">
                    Cảm xúc ngày {date.toLocaleDateString("vi-VN")}
                  </h4>
                  
                  <div className="grid grid-cols-5 gap-2 mt-4">
                    {moods.map((mood) => (
                      <Button
                        key={mood.value}
                        variant={selectedMood === mood.value ? "default" : "outline"}
                        className="flex flex-col items-center p-3 h-auto"
                        onClick={() => setSelectedMood(mood.value)}
                      >
                        {mood.icon}
                        <span className="mt-2 text-xs">{mood.label}</span>
                      </Button>
                    ))}
                  </div>
                  
                  <div className="mt-4">
                    <textarea
                      className="w-full p-3 rounded-md border min-h-[100px]"
                      placeholder="Ghi chú về cảm xúc hôm nay (không bắt buộc)"
                    />
                  </div>
                  
                  <div className="mt-4 flex justify-end">
                    <Button className="thrive-button">Lưu cảm xúc</Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="stats" className="p-0 pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Thống kê tâm trạng</CardTitle>
              <CardDescription>
                Phân tích tâm trạng trong tháng
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="thrive-card text-center p-6">
                  <p className="text-muted-foreground mb-2">Trung bình</p>
                  <h3 className="text-3xl font-bold">7.2</h3>
                  <p className="text-sm text-muted-foreground mt-2">Trong tháng 4</p>
                </div>
                
                <div className="thrive-card text-center p-6">
                  <p className="text-muted-foreground mb-2">Ngày tốt nhất</p>
                  <h3 className="text-3xl font-bold">9.5</h3>
                  <p className="text-sm text-muted-foreground mt-2">08/04/2025</p>
                </div>
                
                <div className="thrive-card text-center p-6">
                  <p className="text-muted-foreground mb-2">Ngày thấp nhất</p>
                  <h3 className="text-3xl font-bold">4.0</h3>
                  <p className="text-sm text-muted-foreground mt-2">02/04/2025</p>
                </div>
              </div>
              
              <div className="mt-6">
                <h4 className="font-medium mb-3">Gợi ý từ Ami</h4>
                <div className="thrive-card border border-thrive-lavender p-4">
                  <p className="text-muted-foreground">
                    Tâm trạng của bạn thường tốt hơn vào các ngày cuối tuần. Hãy thử đưa hoạt động mà bạn thích vào các ngày đầu tuần để cân bằng tâm trạng.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MoodTracker;
