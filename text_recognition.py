import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Parlez quelque chose...")
        audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio, language="fr-FR") 
        print("Vous avez dit :", text)
        return text

speech_to_text()
