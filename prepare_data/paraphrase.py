import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ParaphraseMachine:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        

    def paraphrase(self, input_sentence, num_of_sentences = 5):
        encoding = self.tokenizer.encode_plus(input_sentence, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)


        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            repetition_penalty=0.8, # adjust this as we see fit to avoid repetition in sentences
            num_return_sequences=num_of_sentences
        )

        output_list = []
        for output in outputs:
            decoded_sentence = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_list.append(decoded_sentence)

        return output_list
        

if __name__ == "__main__":
    pm = ParaphraseMachine()
    
    input_sentence = "This is fine! This is exactly the reason this chatbot is created! We aim to develop compassion for you! Let us begin! "
    #input_sentence =  "paraphrase: " + input_sentence + " </s>"
    
    #generated_sentence = pm.batch_paraphrase(input_sentence, 5)
    generated_sentence = pm.paraphrase(input_sentence, num_of_sentences=5)
    print(generated_sentence)