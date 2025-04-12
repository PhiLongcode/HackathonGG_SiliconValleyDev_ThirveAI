
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

interface SuggestedQuestionsProps {
  questions: string[];
  onSelect: (question: string) => void;
}

const SuggestedQuestions = ({ questions, onSelect }: SuggestedQuestionsProps) => {
  return (
    <motion.div 
      className="px-4 py-3 border-t flex flex-wrap gap-2 overflow-x-auto no-scrollbar"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {questions.map((question, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
        >
          <Button 
            variant="outline" 
            size="sm" 
            className="whitespace-nowrap rounded-full border-primary/20 hover:bg-primary/10 hover:text-primary"
            onClick={() => onSelect(question)}
          >
            {question}
          </Button>
        </motion.div>
      ))}
    </motion.div>
  );
};

export default SuggestedQuestions;
