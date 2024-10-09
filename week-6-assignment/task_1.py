# from datasets import load_dataset

# dataset = load_dataset("qanastek/MASSIVE")
# print(dataset)


# # Define the list of desired locales
# locales = ["af-ZA", "da-DK", "de-DE", "en-US", "es-ES", "fr-FR", "fi-FI", "hu-HU", "is-IS", "it-IT", "jv-ID", "lv-LV", "ms-MY", "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SL", "sv-SE", "sq-AL", "sw-KE", "tl-PH", "tr-TR", "vi-VN", "cy-GB"]

# # Filter the dataset to include only the desired locales and columns
# filtered_dataset = dataset.filter(lambda example: example['locale'] in locales).map(lambda example: {k: example[k] for k in ['locale', 'partition', 'utt', 'tokens']})

# print(filtered_dataset)

# from datasets import load_dataset
# from google.colab import drive

# # Mount Google Drive
# drive.mount('/content/drive')

# # Specify the destination folder in Google Drive
# drive_folder = '/content/drive/My Drive/assignment_data/dal/assign6/locale_files'  # Change to your desired folder

# import os
# # Assuming 'drive_folder' is defined as in the previous response

# os.makedirs(drive_folder, exist_ok=True)  # Create directories if they don't exist

# # Save the dataset to a local file in your Drive
# filtered_dataset.save_to_disk(os.path.join(drive_folder,'filtered_data'))