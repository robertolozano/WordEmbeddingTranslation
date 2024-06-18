import numpy as np
import fasttext
import fasttext.util
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import orthogonal_procrustes
from deep_translator import GoogleTranslator
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
g
#README: word embeddings must be in same file as final_project.propert
#IMPORTANT: BlEU_SCORE LIBRARY IS ONLY SUPPORTED IN PYTHON 3.11 AND LOWER!!!
#INCLUDED IN ZIP FILE ARE .GZ files, PLEASE UNCOMPRESS THOSE TO GET .bin files

fasttext.util.download_model('en', if_exists='ignore')
english_model = fasttext.load_model('cc.en.300.bin')

fasttext.util.download_model('es', if_exists='ignore')
spanish_model = fasttext.load_model('cc.es.300.bin')

training_bilingual_dict = {
    'pain': 'dolor',
    'fever': 'fiebre',
    'cough': 'tos',
    'cold': 'resfriado',
    'flu': 'gripe',
    'infection': 'infección',
    'virus': 'virus',
    'bacteria': 'bacteria',
    'allergy': 'alergia',
    'nausea': 'náusea',
    'vomiting': 'vómitos',
    'diarrhea': 'diarrea',
    'constipation': 'estreñimiento',
    'dizziness': 'mareo',
    'fatigue': 'fatiga',
    'insomnia': 'insomnio',
    'asthma': 'asma',
    'bronchitis': 'bronquitis',
    'pneumonia': 'neumonía',
    'tuberculosis': 'tuberculosis',
    'hypertension': 'hipertensión',
    'diabetes': 'diabetes',
    'anemia': 'anemia',
    'cancer': 'cáncer',
    'tumor': 'tumor',
    'hypertension': 'hipertensión',
    'cholesterol': 'colesterol',
    'osteoporosis': 'osteoporosis',
    'fracture': 'fractura',
    'sprain': 'esguince',
    'burn': 'quemadura',
    'cut': 'corte',
    'wound': 'herida',
    'bruise': 'moretón',
    'swelling': 'hinchazón',
    'inflammation': 'inflamación',
    'surgery': 'cirugía',
    'operation': 'operación',
    'recovery': 'recuperación',
    'treatment': 'tratamiento',
    'medication': 'medicación',
    'therapy': 'terapia',
    'vaccine': 'vacuna',
    'immunization': 'inmunización',
    'diagnosis': 'diagnóstico',
    'symptom': 'síntoma',
    'disease': 'enfermedad',
    'disorder': 'trastorno',
    'condition': 'condición',
    'patient': 'paciente',
    'doctor': 'doctor',
    'nurse': 'enfermera',
    'clinic': 'clínica',
    'hospital': 'hospital',
    'emergency': 'emergencia',
    'ambulance': 'ambulancia',
    'pharmacy': 'farmacia',
    'prescription': 'prescripción',
    'appointment': 'cita',
    'check-up': 'chequeo',
    'consultation': 'consulta',
    'procedure': 'procedimiento',
    'x-ray': 'radiografía',
    'scan': 'escáner',
    'biopsy': 'biopsia',
    'ultrasound': 'ultrasonido',
    'EKG': 'electrocardiograma',
    'anesthesia': 'anestesia',
    'abdominal': 'abdominal',
    'antibiotic': 'antibiótico',
    'antiseptic': 'antiséptico',
    'appendicitis': 'apendicitis',
    'bile': 'bilis',
    'biopsy': 'biopsia',
    'bladder': 'vejiga',
    'bone': 'hueso',
    'brain': 'cerebro',
    'breathing': 'respiración',
    'cardiology': 'cardiología',
    'cell': 'célula',
    'chemotherapy': 'quimioterapia',
    'cirrhosis': 'cirrosis',
    'clinic': 'clínica',
    'concussion': 'concusión',
    'contusion': 'contusión',
    'crutch': 'muleta',
    'dehydration': 'deshidratación',
    'dermatology': 'dermatología',
    'diabetes': 'diabetes',
    'diagnosis': 'diagnóstico',
    'dialysis': 'diálisis',
    'disability': 'discapacidad',
    'dislocation': 'dislocación',
    'doctor': 'doctor',
    'dressing': 'vendaje',
    'emergency': 'emergencia',
    'endoscopy': 'endoscopia',
    'epilepsy': 'epilepsia',
    'fainting': 'desmayo',
    'fracture': 'fractura',
    'gastroenterology': 'gastroenterología',
    'glucose': 'glucosa',
    'healing': 'curación',
    'hematology': 'hematología',
    'hernia': 'hernia',
    'hospital': 'hospital',
    'immunology': 'inmunología',
    'incision': 'incisión',
    'infection': 'infección',
    'injection': 'inyección',
    'insulin': 'insulina',
    'intestine': 'intestino',
    'IV': 'intravenoso',
    'larynx': 'laringe',
    'lesion': 'lesión',
    'mammogram': 'mamografía',
    'maternity': 'maternidad',
    'measles': 'sarampión',
    'medication': 'medicación',
    'muscle': 'músculo',
    'nephrology': 'nefrología',
    'neurology': 'neurología',
    'oncology': 'oncología',
    'ophthalmology': 'oftalmología',
    'orthopedics': 'ortopedia',
    'oxygen': 'oxígeno',
    'pancreas': 'páncreas',
    'paralysis': 'parálisis',
    'pharmacology': 'farmacología',
    'physiotherapy': 'fisioterapia',
    'plaster': 'yeso',
    'pneumonia': 'neumonía',
    'polio': 'polio',
    'pregnancy': 'embarazo',
    'psychiatry': 'psiquiatría',
    'pulse': 'pulso',
    'radiology': 'radiología',
    'respiratory': 'respiratorio',
    'rheumatology': 'reumatología',
    'scalpel': 'bisturí',
    'seizure': 'convulsión',
    'skeleton': 'esqueleto',
    'spleen': 'bazo',
    'sprain': 'esguince',
    'suture': 'sutura',
    'tablet': 'comprimido',
    'tendon': 'tendón',
    'therapy': 'terapia',
    'throat': 'garganta',
    'thyroid': 'tiroides',
    'tissue': 'tejido',
    'transplant': 'trasplante',
    'tuberculosis': 'tuberculosis',
    'ulcer': 'úlcera',
    'urine': 'orina',
    'vaccination': 'vacunación',
    'vein': 'vena',
    'virus': 'virus',
    'vitamin': 'vitamina',
    'wound': 'herida',
    'x-ray': 'radiografía',
    'abscess': 'absceso',
    'antiviral': 'antiviral',
    'arthritis': 'artritis',
    'autopsy': 'autopsia',
    'chemotherapy': 'quimioterapia',
    'circulation': 'circulación',
    'congenital': 'congénito',
    'contagious': 'contagioso',
    'cramp': 'calambre',
    'dehydration': 'deshidratación',
    'dermatitis': 'dermatitis',
    'diagnostic': 'diagnóstico',
    'diphtheria': 'difteria',
    'dose': 'dosis',
    'embolism': 'embolia',
    'endocrine': 'endocrino',
    'endoscopy': 'endoscopia',
    'epidural': 'epidural',
    'extraction': 'extracción',
    'fibrosis': 'fibrosis',
    'fracture': 'fractura',
    'gastroenteritis': 'gastroenteritis',
    'gynecology': 'ginecología',
    'hepatitis': 'hepatitis',
    'hormone': 'hormona',
    'hypertension': 'hipertensión',
    'immunodeficiency': 'inmunodeficiencia',
    'inflammation': 'inflamación',
    'intravenous': 'intravenoso',
    'jaundice': 'ictericia',
    'lesion': 'lesión',
    'mammography': 'mamografía',
    'meningitis': 'meningitis',
    'nephrology': 'nefrología',
    'oncology': 'oncología',
    'ophthalmology': 'oftalmología',
    'orthopedics': 'ortopedia',
    'osteoporosis': 'osteoporosis',
    'pancreatitis': 'pancreatitis',
    'pathology': 'patología',
    'phlebotomy': 'flebotomía',
    'physiology': 'fisiología',
    'pneumothorax': 'neumotórax',
    'psychosis': 'psicosis',
    'quarantine': 'cuarentena',
    'scoliosis': 'escoliosis',
    'sepsis': 'sepsis',
    'sinusitis': 'sinusitis',
    'thrombosis': 'trombosis',
    'tumor': 'tumor',
    'ulcer': 'úlcera',
    'urethra': 'uretra',
    'vertebrae': 'vértebras',
}

bilingual_dict = {
    'pain': 'dolor',
    'fever': 'fiebre',
    'cough': 'tos',
    'cold': 'resfriado',
    'flu': 'gripe',
    'infection': 'infección',
    'virus': 'virus',
    'bacteria': 'bacteria',
    'allergy': 'alergia',
    'headache': 'dolor de cabeza',
    'nausea': 'náusea',
    'vomiting': 'vómitos',
    'diarrhea': 'diarrea',
    'constipation': 'estreñimiento',
    'dizziness': 'mareo',
    'fatigue': 'fatiga',
    'insomnia': 'insomnio',
    'asthma': 'asma',
    'bronchitis': 'bronquitis',
    'pneumonia': 'neumonía',
    'tuberculosis': 'tuberculosis',
    'hypertension': 'hipertensión',
    'diabetes': 'diabetes',
    'anemia': 'anemia',
    'cancer': 'cáncer',
    'tumor': 'tumor',
    'stroke': 'derrame cerebral',
    'heart attack': 'ataque al corazón',
    'hypertension': 'hipertensión',
    'cholesterol': 'colesterol',
    'osteoporosis': 'osteoporosis',
    'fracture': 'fractura',
    'sprain': 'esguince',
    'burn': 'quemadura',
    'cut': 'corte',
    'wound': 'herida',
    'bruise': 'moretón',
    'swelling': 'hinchazón',
    'inflammation': 'inflamación',
    'surgery': 'cirugía',
    'operation': 'operación',
    'recovery': 'recuperación',
    'treatment': 'tratamiento',
    'medication': 'medicación',
    'therapy': 'terapia',
    'vaccine': 'vacuna',
    'immunization': 'inmunización',
    'diagnosis': 'diagnóstico',
    'symptom': 'síntoma',
    'disease': 'enfermedad',
    'disorder': 'trastorno',
    'condition': 'condición',
    'patient': 'paciente',
    'doctor': 'doctor',
    'nurse': 'enfermera',
    'clinic': 'clínica',
    'hospital': 'hospital',
    'emergency': 'emergencia',
    'ambulance': 'ambulancia',
    'pharmacy': 'farmacia',
    'prescription': 'prescripción',
    'appointment': 'cita',
    'check-up': 'chequeo',
    'consultation': 'consulta',
    'procedure': 'procedimiento',
    'x-ray': 'radiografía',
    'scan': 'escáner',
    'biopsy': 'biopsia',
    'ultrasound': 'ultrasonido',
    'MRI': 'resonancia magnética',
    'CT scan': 'tomografía computarizada',
    'EKG': 'electrocardiograma',
    'anesthesia': 'anestesia',
    'ICU': 'UCI (Unidad de Cuidados Intensivos)',
    'ER': 'sala de emergencias',
    'abdominal': 'abdominal',
    'antibiotic': 'antibiótico',
    'antiseptic': 'antiséptico',
    'appendicitis': 'apendicitis',
    'bile': 'bilis',
    'biopsy': 'biopsia',
    'bladder': 'vejiga',
    'blood pressure': 'presión arterial',
    'bone': 'hueso',
    'brain': 'cerebro',
    'breathing': 'respiración',
    'cardiology': 'cardiología',
    'cell': 'célula',
    'chemotherapy': 'quimioterapia',
    'cirrhosis': 'cirrosis',
    'clinic': 'clínica',
    'concussion': 'concusión',
    'contusion': 'contusión',
    'crutch': 'muleta',
    'dehydration': 'deshidratación',
    'dermatology': 'dermatología',
    'diabetes': 'diabetes',
    'diagnosis': 'diagnóstico',
    'dialysis': 'diálisis',
    'disability': 'discapacidad',
    'dislocation': 'dislocación',
    'doctor': 'doctor',
    'dressing': 'vendaje',
    'emergency': 'emergencia',
    'endoscopy': 'endoscopia',
    'epilepsy': 'epilepsia',
    'fainting': 'desmayo',
    'fracture': 'fractura',
    'gallbladder': 'vesícula biliar',
    'gastroenterology': 'gastroenterología',
    'glucose': 'glucosa',
    'healing': 'curación',
    'hematology': 'hematología',
    'hepatitis': 'hepatitis',
    'hernia': 'hernia',
    'hospital': 'hospital',
    'immunology': 'inmunología',
    'incision': 'incisión',
    'infection': 'infección',
    'injection': 'inyección',
    'insulin': 'insulina',
    'intestine': 'intestino',
    'IV': 'intravenoso',
    'larynx': 'laringe',
    'lesion': 'lesión',
    'mammogram': 'mamografía',
    'maternity': 'maternidad',
    'measles': 'sarampión',
    'medical history': 'historial médico',
    'medication': 'medicación',
    'muscle': 'músculo',
    'nephrology': 'nefrología',
    'neurology': 'neurología',
    'oncology': 'oncología',
    'ophthalmology': 'oftalmología',
    'orthopedics': 'ortopedia',
    'oxygen': 'oxígeno',
    'pancreas': 'páncreas',
    'paralysis': 'parálisis',
    'pharmacology': 'farmacología',
    'physiotherapy': 'fisioterapia',
    'plaster': 'yeso',
    'pneumonia': 'neumonía',
    'polio': 'polio',
    'pregnancy': 'embarazo',
    'psychiatry': 'psiquiatría',
    'pulse': 'pulso',
    'radiology': 'radiología',
    'respiratory': 'respiratorio',
    'rheumatology': 'reumatología',
    'scalpel': 'bisturí',
    'seizure': 'convulsión',
    'skeleton': 'esqueleto',
    'spleen': 'bazo',
    'sprain': 'esguince',
    'suture': 'sutura',
    'tablet': 'comprimido',
    'tendon': 'tendón',
    'therapy': 'terapia',
    'throat': 'garganta',
    'thyroid': 'tiroides',
    'tissue': 'tejido',
    'transplant': 'trasplante',
    'tuberculosis': 'tuberculosis',
    'ulcer': 'úlcera',
    'urine': 'orina',
    'vaccination': 'vacunación',
    'vein': 'vena',
    'virus': 'virus',
    'vitamin': 'vitamina',
    'wound': 'herida',
    'x-ray': 'radiografía',
    'abscess': 'absceso',
    'antiviral': 'antiviral',
    'arthritis': 'artritis',
    'autopsy': 'autopsia',
    'chemotherapy': 'quimioterapia',
    'circulation': 'circulación',
    'congenital': 'congénito',
    'contagious': 'contagioso',
    'coronary artery': 'arteria coronaria',
    'cramp': 'calambre',
    'dehydration': 'deshidratación',
    'dermatitis': 'dermatitis',
    'diagnostic': 'diagnóstico',
    'diphtheria': 'difteria',
    'dose': 'dosis',
    'embolism': 'embolia',
    'endocrine': 'endocrino',
    'endoscopy': 'endoscopia',
    'epidural': 'epidural',
    'extraction': 'extracción',
    'fibrosis': 'fibrosis',
    'fracture': 'fractura',
    'gallstone': 'cálculo biliar',
    'gastroenteritis': 'gastroenteritis',
    'gynecology': 'ginecología',
    'hormone': 'hormona',
    'hypertension': 'hipertensión',
    'immunodeficiency': 'inmunodeficiencia',
    'inflammation': 'inflamación',
    'intravenous': 'intravenoso',
    'jaundice': 'ictericia',
    'kidney stone': 'cálculo renal',
    'lesion': 'lesión',
    'mammography': 'mamografía',
    'meningitis': 'meningitis',
    'nephrology': 'nefrología',
    'oncology': 'oncología',
    'ophthalmology': 'oftalmología',
    'orthopedics': 'ortopedia',
    'osteoporosis': 'osteoporosis',
    'pancreatitis': 'pancreatitis',
    'pathology': 'patología',
    'phlebotomy': 'flebotomía',
    'physiology': 'fisiología',
    'pneumothorax': 'neumotórax',
    'psychosis': 'psicosis',
    'quarantine': 'cuarentena',
    'renal failure': 'insuficiencia renal',
    'rheumatoid arthritis': 'artritis reumatoide',
    'scoliosis': 'escoliosis',
    'sepsis': 'sepsis',
    'sinusitis': 'sinusitis',
    'thrombosis': 'trombosis',
    'tumor': 'tumor',
    'ulcer': 'úlcera',
    'urethra': 'uretra',
    'varicose veins': 'várices',
    'vertebrae': 'vértebras',
    'stitches': 'sutura',
    'fainting':'desmayo',
    'itching': 'picor',
    'numbness': 'entumecimiento',
    'tendon': 'tendón',
    'swelling': 'hinchazón',
    'rash': 'salpullido',
    'bruise': 'moretón',
    'stroke': 'infarto',
    'malnutrition': 'desnutrición',
    'seizure': 'convulsión',
    'ultrasound': 'ultrasonido',
    'biopsy': 'biopsia',
    'sprain': 'esguince'
}

english_vectors = []
spanish_vectors = []

for eng_word, spa_word in training_bilingual_dict.items():
    if eng_word in english_model.get_words() and spa_word in spanish_model.get_words():
        english_vectors.append(english_model.get_word_vector(eng_word))
        spanish_vectors.append(spanish_model.get_word_vector(spa_word))
    else:
        if eng_word not in english_model.get_words():
            print(f"English word '{eng_word}' not found in the model.")
        if spa_word not in spanish_model.get_words():
            print(f"Spanish word '{spa_word}' not found in the model.")

english_vectors = np.array(english_vectors)
spanish_vectors = np.array(spanish_vectors)

assert english_vectors.shape == spanish_vectors.shape

R, _ = orthogonal_procrustes(english_vectors, spanish_vectors)

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

english_vectors_aligned = normalize(np.dot(english_vectors, R))
spanish_vectors_normalized = normalize(spanish_vectors)

spanish_words = spanish_model.get_words()
spanish_word_vectors = np.array([spanish_model.get_word_vector(word) for word in spanish_words])
spanish_word_vectors_normalized = normalize(spanish_word_vectors)
nearest_neighbors = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(spanish_word_vectors_normalized)

def translate_word(word, source_model, target_words, nearest_neighbors, R, k=5):
    if word not in source_model.get_words():
        print(f"Word '{word}' not found in the source model.")
        return None, []
    source_vector = source_model.get_word_vector(word)
    source_vector_aligned = normalize(np.dot(source_vector, R).reshape(1, -1))
    
    distances, indices = nearest_neighbors.kneighbors(source_vector_aligned, n_neighbors=k)
    neighbors = [(target_words[idx], dist) for dist, idx in zip(distances[0], indices[0])]
    top_word = neighbors[0][0] if neighbors else None
    return top_word, neighbors

def dictionary_lookup(word, bilingual_dict):
    return bilingual_dict.get(word, "Not found")

def google_translate(word, src='en', dest='es'):
    try:
        translation = GoogleTranslator(source=src, target=dest).translate(word)
        return translation
    except Exception as e:
        print(f"Error translating {word}: {e}")
        return "Translation error"

def calculate_bleu(reference, hypothesis):
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing_function)

def plot_vectors(vectors_1, vectors_2, labels, filename, title):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    vectors_2d = tsne.fit_transform(np.vstack((vectors_1, vectors_2)))
    
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:len(vectors_1), 0], vectors_2d[:len(vectors_1), 1], c='r', label='English Vectors')
    plt.scatter(vectors_2d[len(vectors_1):, 0], vectors_2d[len(vectors_1):, 1], c='b', label='Spanish Vectors')
    
    for i in range(len(labels)):
        plt.text(vectors_2d[i, 0], vectors_2d[i, 1], labels[i], fontsize=9)
    
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.close()

plot_vectors(english_vectors, spanish_vectors, list(training_bilingual_dict.keys()), 'initial_vector_spaces.png', "Initial Vector Spaces")

plot_vectors(english_vectors_aligned, spanish_vectors_normalized, list(training_bilingual_dict.keys()), 'aligned_vector_spaces.png', "Aligned Vector Spaces")

test_words = list(bilingual_dict.keys())

bleu_scores_dict = []
bleu_scores_google = []
bleu_scores_improved = []

for word in test_words:
    true_translation = bilingual_dict[word]
    
    print(f'-----------------------------------------')
    print(f'Translating "{word}"')
    
    dict_translation = dictionary_lookup(word, bilingual_dict)
    print(f'Bilingual dictionary translation of "{word}" is: {dict_translation}')
    
    google_translation = google_translate(word)
    print(f'Google Translate translation of "{word}" is: {google_translation}')
    
    top_word, translated_neighbors = translate_word(word, english_model, spanish_words, nearest_neighbors, R, k=5)
    print(f'Vector Space Mapping Translation of "{word}" is: {top_word}')
    
    bleu_dict = calculate_bleu(true_translation, dict_translation if dict_translation != "Not found" else "")
    bleu_google = calculate_bleu(true_translation, google_translation if google_translation != "Translation error" else "")
    bleu_improved = calculate_bleu(true_translation, top_word if top_word else "")
    
    bleu_scores_dict.append(bleu_dict)
    bleu_scores_google.append(bleu_google)
    bleu_scores_improved.append(bleu_improved)

avg_bleu_dict = np.mean(bleu_scores_dict)
avg_bleu_google = np.mean(bleu_scores_google)
avg_bleu_improved = np.mean(bleu_scores_improved)

print(f'-----------------------------------------')
print(f'Average BLEU score for dictionary translation: {avg_bleu_dict}')
print(f'Average BLEU score for Google Translate translation: {avg_bleu_google}')
print(f'Average BLEU score for improved method translation: {avg_bleu_improved}')
