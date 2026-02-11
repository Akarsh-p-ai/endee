from sentence_transformers import SentenceTransformer
from vector_db import EndeeVectorDB
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents (your "database")
documents = [
    # Comprehensive Spiritual Guide of India

## Section 1: Karnataka Spiritual Heritage
Focus on the Seven Muktisthalas and major centers of South Indian architecture.

### 1. Gokarna Mahabaleshwar
* *Location:* Gokarna, Uttara Kannada
* *Deity:* Lord Shiva (Atma Linga)
* *Significance:* Known as "Dakshina Kashi." It is the only place where the Atma Linga of Lord Shiva is located.
* *Key Feature:* Devotees must take a holy dip in the Arabian Sea at Om Beach or Koti Teertha before entering.

### 2. Udupi Sri Krishna Mutt
* *Location:* Udupi
* *Deity:* Lord Krishna (Bala Krishna)
* *Key Feature:* The deity is viewed through the 'Kanakana Kindi'—a silver-plated window with nine holes.

### 3. Sringeri Sharada Peetham
* *Location:* Sringeri, Chikkamagaluru
* *Deity:* Goddess Sharadamba
* *Significance:* The first of four monasteries established by Adi Shankara in the 8th century.

### 4. Murudeshwar Temple
* *Location:* Murudeshwar, Uttara Kannada
* *Key Feature:* Features the world’s second-tallest Shiva statue (123 ft) and a 20-story Raja Gopura overlooking the sea.

### 5. Dharmasthala Manjunatha Temple
* *Location:* Dharmasthala, Dakshina Kannada
* *Significance:* A unique confluence where the deity is Shiva, the priests are Vaishnavite, and the administration is Jain.

### 6. Shravanabelagola (Gommateshwara)
* *Location:* Hassan District
* *Faith:* Jainism
* *Key Feature:* A 57-foot monolithic statue of Lord Bahubali, one of the largest free-standing statues in the world.

---

## Section 2: The 12 Jyotirlingas of Lord Shiva
The 12 most sacred shrines where Shiva is worshipped as a pillar of light.

1. *Somnath (Gujarat):* Located in Prabhas Patan; the first of the twelve Jyotirlingas.
2. *Mallikarjuna (Andhra Pradesh):* Located on Srisailam Mountain; also recognized as a Shakti Peeth.
3. *Mahakaleshwar (Madhya Pradesh):* Ujjain; features the unique South-facing (Dakshinamurthi) Linga.
4. *Omkareshwar (Madhya Pradesh):* Located on an island shaped like the sacred "Om" symbol.
5. *Kedarnath (Uttarakhand):* Situated in the Himalayas; the highest and most remote Jyotirlinga.
6. *Bhimashankar (Maharashtra):* Located in the Sahyadri hills; the source of the Bhima River.
7. *Kashi Vishwanath (Uttar Pradesh):* Located in Varanasi, the spiritual heart of India.
8. *Trimbakeshwar (Maharashtra):* Near Nashik; the source of the Godavari River.
9. *Baidyanath (Jharkhand):* Deoghar; famous for the Shravani Mela.
10. *Nageshwar (Gujarat):* Near Dwarka; believed to protect from all types of poisons.
11. *Ramanathaswamy (Tamil Nadu):* Rameshwaram; where Lord Rama worshipped Shiva.
12. *Grishneshwar (Maharashtra):* Near the UNESCO Ellora Caves.

---

## Section 3: Major Shakti Peethas
Sites where the body parts of Goddess Sati fell; centers of the Divine Feminine.

### 1. Kamakhya Temple
* *Location:* Guwahati, Assam
* *Significance:* Represents the 'Yoni' (creative power); the center of Tantric traditions.

### 2. Kalighat Kali Temple
* *Location:* Kolkata, West Bengal
* *Significance:* Represents the right toes of Sati; a primary pilgrimage for Goddess Kali devotees.

### 3. Chamundeshwari Temple
* *Location:* Mysore, Karnataka
* *Significance:* Located atop Chamundi Hills; represents the hair of the Goddess.

### 4. Ambaji Temple
* *Location:* Gujarat
* *Significance:* Represents the heart of Sati; interestingly, there is no idol, only a sacred Yantra.

### 5. Vishalakshi Temple
* *Location:* Varanasi, Uttar Pradesh
* *Significance:* Represents the earrings of Sati; located very close to the Kashi Vishwanath temple.
*
]

# Convert documents to embeddings
embeddings = model.encode(documents)

print("\n✅ Documents converted to embeddings successfully!")

# Connect to Endee and store data
db = EndeeVectorDB()
db.store_embeddings(documents, embeddings)

# Save embeddings locally (so search.py can use them)
np.save("my_ai_project/embeddings.npy", embeddings)
np.save("my_ai_project/texts.npy", documents)

print("✅ Ingestion completed and data stored.")
