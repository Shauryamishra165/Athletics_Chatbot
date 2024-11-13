from create_data_embeddings import user_input
import tkinter as tk


def main():
    response  , docs = user_input("Who won 100 meter Inter IIT men in 2023 and tell time also ?")
    output_text = response.get('output_text', 'No response')
    print(output_text)
    print(docs)

if __name__ == "__main__":
    main()