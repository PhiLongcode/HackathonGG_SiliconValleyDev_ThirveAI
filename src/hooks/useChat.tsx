import { useState, useRef, useEffect } from "react";
import { toast } from "@/components/ui/use-toast";
import { Message, AmiMood } from "@/types/chat";
import { useInactivity } from "@/hooks/useInactivity";
import { GoogleGenerativeAI } from "@google/generative-ai";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "ami",
      text: "Xin chào! Mình là Ami, một chuyên gia tư vấn tâm lý. Bạn đang cảm thấy thế nào hôm nay? Hãy chia sẻ với mình nhé!",
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState("");
  const [amiMood, setAmiMood] = useState<AmiMood>("happy");
  const [isRecording, setIsRecording] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Suggested questions (chỉ liên quan đến tâm lý)
  const suggestedQuestions = [
    "Bạn đang cảm thấy thế nào hôm nay?",
    "Có điều gì đang khiến bạn trăn trở không?",
    "Bạn muốn chia sẻ gì về cảm xúc của mình?",
    "Hôm nay bạn đã trải qua điều gì khó khăn chưa?",
  ];

  // Handle inactivity
  const handleInactiveMessage = (amiMessage: Message, mood: AmiMood) => {
    setMessages(prev => [...prev, amiMessage]);
    setAmiMood(mood);
  };

  const { setupInactivityTimer } = useInactivity({
    isEnabled: true,
    onInactiveMessage: handleInactiveMessage
  });

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Khởi tạo Gemini client
  const genAI = new GoogleGenerativeAI('AIzaSyBZ9z29qG2dlFcGFAD-4Ds8Rt_wMM-OZ0U');
  const model = genAI.getGenerativeModel({ model: 'models/gemini-1.5-flash' });

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      text: inputText,
      timestamp: new Date(),
    };
    
    setMessages([...messages, userMessage]);
    setInputText("");
    setAmiMood("thinking");
    setIsTyping(true);

    setTimeout(async () => {
      try {
        const prompt = `
          Bạn là một chuyên gia tư vấn tâm lý chuyên nghiệp. Hãy trả lời với phong cách thân thiện và đồng cảm.

          Quy tắc định dạng câu trả lời:
          1. Bắt đầu với emoji phù hợp và lời chào ngắn gọn, thân thiện
          2. Chia nội dung thành các đoạn ngắn (2-3 dòng)
          3. Mỗi đoạn cách nhau bằng 1 dòng trống
          4. Với danh sách:
             • Thụt lề 3 khoảng trắng
             • Mỗi điểm bắt đầu bằng emoji thích hợp
             • Các điểm cách nhau 1 dòng
          5. Với lời khuyên:
             • Đánh số thứ tự rõ ràng
             • Mỗi bước có emoji riêng
             • Giải thích ngắn gọn, dễ hiểu
          6. Kết thúc bằng:
             • Câu động viên ngắn gọn
             • Emoji tích cực
             • Lời chào thân thiện

          Nếu người dùng hỏi về chủ đề khác:
          "Mình là chuyên gia tư vấn tâm lý, nên mình chỉ có thể giúp bạn với các vấn đề về cảm xúc và tinh thần.

          Bạn đang cảm thấy thế nào, mình có thể hỗ trợ gì cho bạn? 🤗"

          Tin nhắn của người dùng: "${inputText}"
        `;
        
        const result = await model.generateContent(prompt);
        const response = await result.response;
        let responseText = response.text();

        // Cải thiện format văn bản
        responseText = responseText
          // Xử lý emoji và dấu câu
          .replace(/([\u{1F300}-\u{1F9FF}])/gu, '\n$1') // Tách emoji ra dòng mới
          .replace(/([.!?])\s*/g, '$1\n\n') // Thêm dòng trống sau dấu câu
          .replace(/•/g, '\n\n   • ') // Định dạng bullet points với thụt lề
          .replace(/(\d+\.)/g, '\n\n$1') // Định dạng số thứ tự
          .replace(/:\s*/g, ':\n') // Xuống dòng sau dấu hai chấm
          .replace(/\n{3,}/g, '\n\n') // Chuẩn hóa khoảng trống
          .replace(/^\s+|\s+$/gm, '') // Xóa khoảng trắng thừa
          .replace(/(?<=\n)(\s*[-•])/g, '   $1') // Thụt lề cho danh sách
          .trim();

        // Thêm khoảng cách cho các đoạn văn
        responseText = responseText
          .split('\n')
          .filter(line => line.trim() !== '')
          .join('\n\n');

        const amiMessage: Message = {
          id: Date.now().toString(),
          type: "ami",
          text: responseText,
          timestamp: new Date(),
        };

        setIsTyping(false);
        setAmiMood("happy");
        setIsSpeaking(true);
        setMessages(prev => [...prev, amiMessage]);

        setTimeout(() => {
          setIsSpeaking(false);
        }, Math.min(responseText.length * 40, 6000)); // Reduced max duration to 6 seconds

      } catch (error) {
        console.error("API Error:", error);
        setIsTyping(false);
        setAmiMood("sad");
        toast({
          title: "Lỗi kết nối",
          description: "Không thể kết nối tới AI. Vui lòng thử lại!",
        });
      }
    }, 1500);
  };

  const handleToggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      toast({
        title: "Voice-to-Text",
        description: "Đã chuyển giọng nói thành văn bản"
      });
      // Giả lập kết quả voice-to-text với nội dung tâm lý
      setInputText("Hôm nay tôi cảm thấy hơi buồn và căng thẳng, không biết làm sao để giải tỏa.");
    } else {
      setIsRecording(true);
      toast({
        title: "Đang ghi âm",
        description: "Hãy chia sẻ cảm xúc của bạn...",
      });
    }
  };

  const handleSuggestedQuestion = (question: string) => {
    setInputText(question);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  return {
    messages,
    inputText,
    setInputText,
    amiMood,
    isRecording,
    isTyping,
    isSpeaking,
    setIsSpeaking,
    suggestedQuestions,
    messagesEndRef,
    inputRef,
    handleSendMessage,
    handleToggleRecording,
    handleSuggestedQuestion,
    setupInactivityTimer,
  };
}