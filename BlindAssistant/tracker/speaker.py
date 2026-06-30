import sys
import pyttsx3

def main():
    if len(sys.argv) < 2:
        return
    text = sys.argv[1]
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    main()
