def word_inclusion_criteria(word):
    if len(word) < 3:
        return False

    try:
        float(word.replace(',',''))
        return False
    except:
        pass

    return True


