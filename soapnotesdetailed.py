import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from time import time


class AzureOpenAIWrapper:
    """
    A wrapper class for Azure OpenAI interaction, promoting code readability
    and maintainability.
    """

    def __init__(self):
        """
        Loads environment variables for API configuration securely.
        """
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.api_version = os.getenv("API_VERSION")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_model_deployment = os.getenv("AZURE_MODEL_DEPLOYMENT")

    def create_chat_completion(self, prompt, max_tokens=3000, temperature=0.2, top_p=0.95):
        """
        Performs chat completion using the configured Azure OpenAI model.

        Args:
            prompt (str): The prompt text for the conversation construction.
            max_tokens (int, optional): The maximum number of tokens allowed in the response. Defaults to 15000.
            temperature (float, optional): Controls the randomness of the generated text. Defaults to 0.2.
            top_p (float, optional): The probability of picking the top words in the vocabulary. Defaults to 0.95.

        Returns:
            str: The generated conversation reconstruction.
        """

        client = AzureOpenAI(api_key=self.api_key,
                             api_version=self.api_version,
                             azure_endpoint=self.azure_endpoint)
        start_time = time()
        response = client.chat.completions.create(
            model=self.azure_model_deployment,  # Replace with the appropriate Azure OpenAI engine
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "system", "content": "Assistant is a conversation constructor between the doctor and "
                                                    "patient."},
                      {"role": "user", "content": f"{prompt}"}]
        )

        total_time = time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"The time difference is: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
        print(f"The number of tokens being used are {response.usage.total_tokens}")
        return response.choices[0].message.content


def summarize_text(json_transcription):
    text = json_transcription["conversation"]
    prompt = (
              f"""
              You are an AI assistant to a doctor. Your task is to interpret the spoken conversation {text} 
              between the doctor and a patient during a "geriatric assessment" consultation.   
Your task is to consolidate all the available health information regarding the patient (including any available past 
consultations, chatbot interactions, reports uploaded, Electronic Health records available etc) 
with the information gathered during the latest consultation into one comprehensive view of the patient health.  
 

Assimilate all the information collected so far into a detailed SOAP note that any clinician can use to quickly 
understand the patient’s history and the current conversation.  

Include the following sections in the detailed SOAP note:  

Subjective: In succinct bullet points, provide the following,  if available: 

    List of current symptoms reported by the patient  (e.g., "Cough", "Fatigue", "Nausea"). 

    Timeline of medical history (e.g., "Hypertension diagnosed 5 years ago"). 

    Family history of relevant conditions (e.g., "Father with diabetes"). 

    Social history including smoking, alcohol use, and occupational hazards (e.g., "Non-smoker", "Social drinker", "Works in a construction environment"). 

Objective: In succinct bullet points, include objective details as under, if available: 

    Vital signs (e.g., "Blood pressure: 140/90 mmHg", "Heart rate: 88 bpm", "Temperature: 37.5°C"). 

    Physical exam findings using standardized medical terminology (e.g., "Normocephalic, atraumatic", "Lungs clear to auscultation bilaterally"). 

    Results of any diagnostic tests mentioned during the consultation (e.g., "Urinalysis negative for leukocytes"). 

Assessment: In succinct bullet points, list out the following, if available: 

    Formulate a preliminary diagnosis or a list of potential diagnoses based on the subjective and objective data. (e.g., "Possible upper respiratory infection", "Differential diagnoses include pneumonia and bronchitis"). 

    Briefly explain the reasoning behind the diagnosis/diagnoses using appropriate medical terminology (e.g., "Acute onset of cough and fatigue suggests upper respiratory infection. Negative urinalysis argues against urinary tract infection"). 

Plan: In succinct bullet points, list out suggested treatment plan for the patient, if available: 

    Recommendations for further tests if needed (e.g., "Chest X-ray to rule out pneumonia"). 

    Explanation of prescribed medications or treatments (e.g., "Amoxicillin 500mg three times daily for 7 days"). 
    Check for any adverse reactions of medications. 

    Instructions for the patient regarding self-care and follow-up (e.g., "Increase fluid intake", "Schedule a follow-up
     appointment in 3 days if symptoms worsen"). 

 
Add a section on detailed references to evidence-based medical research literature that would be relevant to the 
symptoms that the patient has experienced. 

Maintain patient confidentiality by avoiding any personal details beyond what's necessary for medical documentation. 

After you have generated a summary, check back with the original input text to confirm that the generated summary 
accurately preserves the facts presented or discussed in the consultation.  

Repeat this validation twice to ensure the AI is not hallucinating. 
              """

              )
    client = AzureOpenAIWrapper()  # Use the wrapper class for clean interaction
    output = client.create_chat_completion(prompt)

    summarised_output = {"conversation": output}
    return summarised_output


if __name__ == '__main__':
    with open("input_file.json", "r") as f:
        json_text = json.load(f)
    summary = summarize_text(json_text)
    print(summary)
