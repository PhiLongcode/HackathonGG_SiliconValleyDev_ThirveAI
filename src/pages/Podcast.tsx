
import { useState } from "react";
import { Headphones, Play, Search, Filter, Clock, Heart, Download, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useIsMobile } from "@/hooks/use-mobile";

interface MediaItem {
  id: string;
  title: string;
  author: string;
  duration: string;
  type: "podcast" | "video";
  thumbnail: string;
  category: string;
  isFeatured: boolean;
}

const mockMedia: MediaItem[] = [
  {
    id: "1",
    title: "Cách vượt qua những ngày tồi tệ",
    author: "Tiến sĩ Nguyễn Văn A",
    duration: "15:30",
    type: "podcast",
    thumbnail: "podcast1.jpg",
    category: "Stress",
    isFeatured: true,
  },
  {
    id: "2",
    title: "Bài tập thở để giảm lo âu",
    author: "Chuyên gia Trần Thị B",
    duration: "08:45",
    type: "video",
    thumbnail: "video1.jpg",
    category: "Lo âu",
    isFeatured: true,
  },
  {
    id: "3",
    title: "Cải thiện giấc ngủ với 5 phút thiền mỗi ngày",
    author: "Tiến sĩ Lê Văn C",
    duration: "21:15",
    type: "podcast",
    thumbnail: "podcast2.jpg",
    category: "Giấc ngủ",
    isFeatured: false,
  },
  {
    id: "4",
    title: "Kỹ thuật thư giãn cơ thể",
    author: "Chuyên gia Phạm Thị D",
    duration: "12:30",
    type: "video",
    thumbnail: "video2.jpg",
    category: "Thư giãn",
    isFeatured: false,
  },
  {
    id: "5",
    title: "Sống tích cực mỗi ngày",
    author: "Tiến sĩ Hoàng Văn E",
    duration: "18:20",
    type: "podcast",
    thumbnail: "podcast3.jpg",
    category: "Tích cực",
    isFeatured: true,
  },
  {
    id: "6",
    title: "Yoga cho người mới bắt đầu",
    author: "Huấn luyện viên Mai Thị F",
    duration: "25:00",
    type: "video",
    thumbnail: "video3.jpg",
    category: "Yoga",
    isFeatured: false,
  },
];

const Podcast = () => {
  const isMobile = useIsMobile();
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  
  const categories = Array.from(new Set(mockMedia.map(item => item.category)));
  
  const filteredMedia = mockMedia.filter(item => {
    const matchesSearch = item.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         item.author.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = activeCategory ? item.category === activeCategory : true;
    
    return matchesSearch && matchesCategory;
  });
  
  const featuredMedia = mockMedia.filter(item => item.isFeatured);
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Podcast & Video</h1>
          <p className="text-muted-foreground">
            Nội dung động lực và hướng dẫn cho sức khỏe tinh thần
          </p>
        </div>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input 
            placeholder="Tìm kiếm podcast và video..." 
            className="pl-9" 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <Button variant="outline" className="flex gap-2">
          <Filter className="h-4 w-4" /> Bộ lọc
        </Button>
      </div>
      
      <div className="flex gap-2 overflow-x-auto pb-2">
        <Button 
          variant={activeCategory === null ? "default" : "outline"}
          size="sm"
          onClick={() => setActiveCategory(null)}
        >
          Tất cả
        </Button>
        
        {categories.map(category => (
          <Button
            key={category}
            variant={activeCategory === category ? "default" : "outline"}
            size="sm"
            onClick={() => setActiveCategory(category)}
          >
            {category}
          </Button>
        ))}
      </div>
      
      <Tabs defaultValue="all" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="all">Tất cả</TabsTrigger>
          <TabsTrigger value="podcast">Podcast</TabsTrigger>
          <TabsTrigger value="video">Video</TabsTrigger>
        </TabsList>
        
        <TabsContent value="all" className="p-0 pt-4">
          {/* Featured section */}
          {!searchTerm && !activeCategory && (
            <section className="mb-8">
              <h2 className="text-xl font-semibold mb-4">Đề xuất cho bạn</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {featuredMedia.map(item => (
                  <MediaCard key={item.id} item={item} />
                ))}
              </div>
            </section>
          )}
          
          {/* All media */}
          <section>
            <h2 className="text-xl font-semibold mb-4">
              {searchTerm || activeCategory ? "Kết quả tìm kiếm" : "Tất cả nội dung"}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredMedia.map(item => (
                <MediaCard key={item.id} item={item} />
              ))}
            </div>
            
            {filteredMedia.length === 0 && (
              <div className="text-center py-12">
                <p className="text-muted-foreground">
                  Không tìm thấy nội dung phù hợp. Hãy thử tìm kiếm khác.
                </p>
              </div>
            )}
          </section>
        </TabsContent>
        
        <TabsContent value="podcast" className="p-0 pt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredMedia
              .filter(item => item.type === "podcast")
              .map(item => (
                <MediaCard key={item.id} item={item} />
              ))}
          </div>
          
          {filteredMedia.filter(item => item.type === "podcast").length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                Không tìm thấy podcast phù hợp.
              </p>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="video" className="p-0 pt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredMedia
              .filter(item => item.type === "video")
              .map(item => (
                <MediaCard key={item.id} item={item} />
              ))}
          </div>
          
          {filteredMedia.filter(item => item.type === "video").length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                Không tìm thấy video phù hợp.
              </p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

interface MediaCardProps {
  item: MediaItem;
}

const MediaCard = ({ item }: MediaCardProps) => {
  return (
    <Card className="overflow-hidden">
      <div className="relative pt-[56.25%] bg-muted">
        <div className="absolute inset-0 flex items-center justify-center">
          {item.type === "podcast" ? (
            <Headphones className="h-12 w-12 text-muted-foreground opacity-50" />
          ) : (
            <Play className="h-12 w-12 text-muted-foreground opacity-50" />
          )}
        </div>
        
        <Button
          className="absolute bottom-3 right-3 rounded-full w-10 h-10 p-0"
          size="icon"
        >
          <Play className="h-5 w-5" />
        </Button>
      </div>
      
      <CardHeader className="p-4 pb-2">
        <div className="flex justify-between items-start gap-2">
          <div>
            <CardTitle className="text-base line-clamp-2">{item.title}</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">{item.author}</p>
          </div>
          <Badge variant="outline">{item.category}</Badge>
        </div>
      </CardHeader>
      
      <CardContent className="p-4 pt-0 pb-2">
        <div className="flex items-center gap-1 text-sm text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>{item.duration}</span>
        </div>
      </CardContent>
      
      <CardFooter className="p-4 pt-0 flex justify-between">
        <div className="flex gap-2">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <Heart className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <Download className="h-4 w-4" />
          </Button>
        </div>
        
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <Share2 className="h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  );
};

export default Podcast;
