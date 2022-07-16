import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer


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
        

class BacktranslationMachine:
    def __init__(self, src="en", tgt="zh"):
        # Languages code: https://developers.google.com/admin-sdk/directory/v1/languages 
        # https://towardsdatascience.com/data-augmentation-in-nlp-using-back-translation-with-marianmt-a8939dfea50a
        self.src = src
        self.tgt = tgt
        
        self.tokenizer1 = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{tgt}")  
        self.model1 = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{tgt}")

        self.tokenizer2 = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt}-{src}")  
        self.model2 = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt}-{src}")


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        

    def process_text(self, lang_code, text):
        formatted_text = [f">>{lang_code}<< {t}" for t in text]
        return formatted_text


    def translation1(self, input_sentence):
        # translate to second language
        formatted_text = self.process_text(self.tgt, input_sentence)
        encoded_lang1 = self.tokenizer1(formatted_text, padding=True, return_tensors="pt")
        translated_encoded_lang1 = self.model1.generate(**encoded_lang1)
        decoded_lang2 = self.tokenizer1.batch_decode(translated_encoded_lang1, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded_lang2

    def translation2(self, input_sentence):
        # translate back to first language
        formatted_text = self.process_text(self.src, input_sentence)
        encoded_lang2 = self.tokenizer2(formatted_text, padding=True, return_tensors="pt")
        print(encoded_lang2)
        translated_encoded_lang1 = self.model2.generate(**encoded_lang2)
        decoded_lang1 = self.tokenizer2.batch_decode(translated_encoded_lang1, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded_lang1

    def backtranslation(self, input_sentence):
        translated_sentence = self.translation1(input_sentence)
        backtranslated_sentence = self.translation2(translated_sentence)
        return backtranslated_sentence

if __name__ == "__main__":

    
    pm = ParaphraseMachine()
    """
    input_sentence = "This is fine! This is exactly the reason this chatbot is created! We aim to develop compassion for you! Let us begin! "
    #input_sentence =  "paraphrase: " + input_sentence + " </s>"
    
    #generated_sentence = pm.batch_paraphrase(input_sentence, 5)
    generated_sentence = pm.paraphrase(input_sentence, num_of_sentences=5)
    print(generated_sentence)
    """

    bm = BacktranslationMachine()
    input_sentence = ["This is fine! This is exactly the reason this chatbot is created! We aim to develop compassion for you! Let us begin! "]
    generated_sentence = bm.backtranslation(input_sentence)
    print(generated_sentence)
    

