import json
import tensorflow as tf
import requests


# Download the ImageNet class index JSON file
response = requests.get('https://storage.googleapis.com/download.tensorflow'
                        '.org/data/imagenet_class_index.json')
imagenet_class_index = response.json()

# Your list of dog breed names from WordNet
dog_breed_names = ['affenpinscher', 'afghan_hound', 'airedale',
                   'american_foxhound', 'american_staffordshire_terrier',
                   'american_water_spaniel', 'appenzeller', 'attack_dog',
                   'australian_terrier', 'basenji', 'basset', 'beagle',
                   'bedlington_terrier', 'belgian_sheepdog',
                   'bernese_mountain_dog', 'bird_dog',
                   'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
                   'bluetick', 'boarhound', 'border_collie', 'border_terrier',
                   'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer',
                   'brabancon_griffon', 'briard', 'brittany_spaniel',
                   'bull_mastiff', 'bulldog', 'bullterrier', 'cairn',
                   'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
                   'chow', 'clumber', 'clydesdale_terrier', 'cocker_spaniel',
                   'collie', 'coondog', 'coonhound', 'corgi', 'courser', 'cur',
                   'curly-coated_retriever', 'dachshund', 'dalmatian',
                   'dandie_dinmont', 'doberman', 'english_foxhound',
                   'english_setter', 'english_springer', 'english_toy_spaniel',
                   'entlebucher', 'eskimo_dog', 'feist', 'field_spaniel',
                   'flat-coated_retriever', 'fox_terrier', 'foxhound',
                   'french_bulldog', 'german_shepherd',
                   'german_short-haired_pointer', 'giant_schnauzer',
                   'golden_retriever', 'gordon_setter', 'great_dane',
                   'great_pyrenees', 'greater_swiss_mountain_dog', 'greyhound',
                   'griffon', 'griffon', 'groenendael', 'guide_dog', 'harrier',
                   'hearing_dog', 'hound', 'housedog', 'hunting_dog',
                   'ibizan_hound', 'irish_setter', 'irish_terrier',
                   'irish_water_spaniel', 'irish_wolfhound',
                   'italian_greyhound', 'japanese_spaniel', 'keeshond',
                   'kelpie', 'kerry_blue_terrier', 'king_charles_spaniel',
                   'komondor', 'kuvasz', 'labrador_retriever',
                   'lakeland_terrier', 'lapdog', 'large_poodle', 'leonberg',
                   'lhasa', 'liver-spotted_dalmatian', 'malamute', 'malinois',
                   'maltese_dog', 'manchester_terrier', 'mastiff',
                   'mexican_hairless', 'miniature_pinscher',
                   'miniature_poodle', 'miniature_schnauzer', 'newfoundland',
                   'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier',
                   'old_english_sheepdog', 'otterhound', 'papillon',
                   'pariah_dog', 'pekinese', 'pembroke', 'pinscher',
                   'plott_hound', 'pointer', 'police_dog', 'pomeranian',
                   'pooch', 'poodle', 'pug', 'puppy', 'rat_terrier', 'redbone',
                   'retriever', 'rhodesian_ridgeback', 'rottweiler',
                   'saint_bernard', 'saluki', 'samoyed', 'sausage_dog',
                   'schipperke', 'schnauzer', 'scotch_terrier',
                   'scottish_deerhound', 'sealyham_terrier', 'seeing_eye_dog',
                   'seizure-alert_dog', 'sennenhunde', 'setter',
                   'shepherd_dog', 'shetland_sheepdog', 'shih-tzu',
                   'siberian_husky', 'silky_terrier', 'skye_terrier',
                   'sled_dog', 'smooth-haired_fox_terrier',
                   'soft-coated_wheaten_terrier', 'spaniel', 'spitz',
                   'sporting_dog', 'springer_spaniel',
                   'staffordshire_bullterrier', 'staghound', 'standard_poodle',
                   'standard_schnauzer', 'sussex_spaniel', 'terrier',
                   'tibetan_mastiff', 'tibetan_terrier', 'toy_dog',
                   'toy_manchester', 'toy_poodle', 'toy_spaniel',
                   'toy_terrier', 'vizsla', 'walker_hound', 'watchdog',
                   'water_dog', 'water_spaniel', 'weimaraner',
                   'welsh_springer_spaniel', 'welsh_terrier',
                   'west_highland_white_terrier', 'whippet',
                   'wire-haired_fox_terrier', 'wirehair', 'wolfhound',
                   'working_dog', 'yorkshire_terrier']

# Normalize breed names to ImageNet format (lowercase, underscores instead
# of spaces)
dog_breed_names_normalized = [name.lower() for name in dog_breed_names]

# Convert the list of class labels to a dictionary mapping class labels to
# wnids
imagenet_labels_to_wnids = {entry[1]: entry[0] for entry in
                            imagenet_class_index.values()}


# Find corresponding wnids in ImageNet for the dog breeds
matched_wnids = []
not_found_breeds = []

for breed in dog_breed_names_normalized:
    if breed in imagenet_labels_to_wnids:
        matched_wnids.append(imagenet_labels_to_wnids[breed])
    else:
        # Capitalize only the first letter of the breed name
        breed_capitalize_first = breed.capitalize()
        if breed_capitalize_first in imagenet_labels_to_wnids:
            matched_wnids.append(
                imagenet_labels_to_wnids[breed_capitalize_first])
        else:
            not_found_breeds.append(breed)

# Print out the matched wnids
print(f"Matched wnids: {matched_wnids}")

# Print out breeds not found
print(f"Breeds not found in ImageNet: {not_found_breeds}")
print(f"Total breeds found: {len(matched_wnids)}")
print(f"Total breeds not found: {len(not_found_breeds)}")
