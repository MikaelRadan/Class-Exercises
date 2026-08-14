#این تابع میانگین نمرات  دانشجو ها رو حساب میکنه 
#جدا از این که شما لیست نمرات را دریافت میکنید . کل نمرارات رو هم به تعداد ان تقسیم میکنه و مقدار میانگین برمیگردونه

def average(scores: list[float]) -> float:
    return sum(scores) / len(scores)

#این تابع میانگین هر دانشجو رو به مشخسات اون اضافه میکنه
#استفاده برای لیست دانشجویان و میانگین نمره هر دانشجو  انجام میده

def add_average_to_students(students: list[dict]) -> None:
    for student in students:
        student["average"] = average(student["scores"])

#در این تابع دانشجویی که بالاترین میانگین  را دارد پیدا میکند 

def top_student(students: list[dict]) -> str:
    best = students[0]
    for student in students:
        if student["average"] > best["average"]:
            best = student
    return best["name"]

#در این تابع دانشجویانی که نمره قبولی گرفته اند را مشخص میکند

def passing_students(students: list[dict], threshold: float = 60) -> list[str]:
  passed = []
  for student in students:
        if student["average"] >= threshold:
            passed.append(student["name"])
  return passed

#در این تابع میانگین کل کلاس محاسبه میشد

def class_average(students: list[dict]) -> float:
    total = 0
    for student in students:
        total += student["average"]
    return total / len(students)

#در این تابع یک گزارش کامل تهیه میکند از وضعیت دانشجویان و ان را چاپ میکند

def print_report(students: list[dict]) -> None:
    for student in students:
        if student["average"] >= 60:
            status = "Pass"
        else:
            status = "Fail"
            print(
            f'{student["name"]}: '
            f'{student["average"]} '
            f'({status})')

#در این تابع منظم ترین  دانشجو از نظر نمرات را پیدا میکند

def most_consistent_student(students: list[dict]) -> str:
    best = students[0]
    best_range = (
        max(best["scores"])
        - min(best["scores"]))
    for student in students:
        current_range = (
            max(student["scores"])
            - min(student["scores"]))
        if current_range < best_range:
            best = student
            best_range = current_range
    return best["name"]


def main() -> None:
    students = [
        {"name": "Sara", "scores": [82, 91, 76]},
        {"name": "Ali", "scores": [65, 70, 72]},
        {"name": "Niloofar", "scores": [95, 89, 92]},
        {"name": "Reza", "scores": [40, 45, 50]},
        {"name": "Maryam", "scores": [50, 55, 57]}]

    add_average_to_students(students)

    print_report(students)

    print()
    print("Class average:", round(class_average(students), 1))
    print("Top student:", top_student(students))
    print("Passing students:", passing_students(students))

    assert round(average([10, 20, 30]), 1) == 20.0
    assert all("average" in s for s in students)
    assert top_student(students) == "Niloofar"
    assert passing_students(students) == ["Sara", "Ali", "Niloofar"]

    other_class = [
        {"name": "Amir", "scores": [30, 20]},
        {"name": "Hana", "scores": [100, 98, 97, 95]},
        {"name": "Kian", "scores": [61]},
        {"name": "Leila", "scores": [45, 50]},]

    add_average_to_students(other_class)

    print()
    print_report(other_class)
    print("Class average:", round(class_average(other_class), 1))
    print("Top student:", top_student(other_class))
    print("Passing students:", passing_students(other_class))

    assert top_student(other_class) == "Hana"
    assert passing_students(other_class) == ["Hana", "Kian"]

    print("\nAll good!")


if __name__ == "__main__":
    main()
