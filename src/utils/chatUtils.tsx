
import { Message, AmiMood } from "@/types/chat";
import { toast } from "@/components/ui/use-toast";
import { ToastAction } from "@/components/ui/toast";
import React from "react";

// Generate encouraging messages for inactive users
export const getEncouragementMessage = (): string => {
  const encouragementMessages = [
    "Bạn vẫn ở đó chứ? Mình luôn sẵn sàng trò chuyện nếu bạn cần.",
    "Đã một lúc rồi nhỉ? Hy vọng bạn vẫn ổn. Mình vẫn ở đây nếu bạn muốn chia sẻ điều gì.",
    "Hôm nay của bạn thế nào? Mình muốn biết thêm về cảm nhận của bạn.",
    "Bạn đang làm gì vậy? Mình rất muốn được trò chuyện cùng bạn!",
  ];
  
  return encouragementMessages[Math.floor(Math.random() * encouragementMessages.length)];
};

// Process user input and generate appropriate Ami response
export const generateAmiResponse = (inputText: string): {
  responseText: string;
  newMood: AmiMood;
  showToast?: () => void;
} => {
  const lowerText = inputText.toLowerCase();
  const hasSadWords = lowerText.includes("buồn") || 
                      lowerText.includes("mệt") || 
                      lowerText.includes("chán") ||
                      lowerText.includes("khó khăn") ||
                      lowerText.includes("không ổn");
  
  if (hasSadWords) {
    const showBreathingExercise = () => {
      setTimeout(() => {
        toast({
          title: "Hít thở sâu",
          description: "Bạn có muốn thực hiện bài tập thở để giảm căng thẳng?",
          action: (
            <ToastAction onClick={handleBreathingExercise} altText="Thử ngay">
              Thử ngay
            </ToastAction>
          ),
        });
      }, 1000);
    };
    
    return {
      responseText: "Mình hiểu cảm giác đó. Hãy chia sẻ thêm để mình có thể giúp bạn tốt hơn. Bạn có muốn thử một số bài tập thở để thư giãn không? 😊",
      newMood: "neutral",
      showToast: showBreathingExercise
    };
  } else if (lowerText.includes("chuyên gia") || lowerText.includes("tư vấn") || lowerText.includes("giúp đỡ")) {
    const showExpertConnect = () => {
      setTimeout(() => {
        toast({
          title: "Kết nối chuyên gia",
          description: "Bạn có muốn gọi video với chuyên gia tâm lý?",
          action: (
            <ToastAction onClick={() => window.location.href = "/video-call"} altText="Gọi ngay">
              Gọi ngay
            </ToastAction>
          ),
        });
      }, 1000);
    };
    
    return {
      responseText: "Bạn muốn kết nối với chuyên gia tâm lý? Mình có thể giúp bạn kết nối ngay bây giờ nếu bạn muốn. Chỉ cần nhấn vào nút 'Gọi chuyên gia' bên dưới nhé.",
      newMood: "excited",
      showToast: showExpertConnect
    };
  } else {
    return {
      responseText: "Cảm ơn bạn đã chia sẻ! Mình luôn ở đây để lắng nghe và hỗ trợ bạn. Bạn có điều gì khác muốn nói với mình không? 😊",
      newMood: "happy"
    };
  }
};

// Handle breathing exercise
export const handleBreathingExercise = () => {
  toast({
    title: "Bài tập hít thở",
    description: "Hít sâu trong 4 giây, giữ 4 giây, thở ra trong 6 giây. Lặp lại 5 lần.",
    duration: 10000,
  });
};
