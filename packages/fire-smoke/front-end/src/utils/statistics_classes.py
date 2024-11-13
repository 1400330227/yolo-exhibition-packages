def statistics_classes(results, names):
    classes = {name: 0 for name in names.values()}
    for result in results:
        for box in result.boxes:
            cls = names[box.cls.item()]
            classes[cls] += 1
    return classes
