# کتابخونه برای ذخیره سازی
import json

# ------------------------------------------
# Book Class
# ------------------------------------------
class Book:
    def __init__(self, title: str, author: str, year: int) -> None:
        self.title = title
        self.author = author
        self.year = year
        self.available = True

    #این تابع برای قرض دادن کتاب استفاده میشه

    def borrow(self) -> bool:
        if self.available:
           self.available = False
           return True
        return False

    #این تابع یرای برگشت کتابها به کتابخونه استفاده میشود

    def return_book(self) -> None:
        self.available = True

    #این تابع اطلاعات کتاب را به مخاطب نمایش میدهد

    def display(self) -> None:
        if self.available:
            status = "Available"
        else:
            status = "Borrowed"

        print(f"{self.title} {self.author} ({self.year}) - {status}")

# ------------------------------------------
# User Class
# ------------------------------------------
class User:

    #این تابع ساخت  کاربری را انجام میدهد

    def __init__(self, name: str) -> None:
        self.name = name
        self.borrowed_books = []

    #این تابع برای قرض گرفتن کتاب توسط کاربر استفاده میشه

    def borrow_book(self, book: Book) -> bool:
        if book.borrow():
            self.borrowed_books.append(book)
            return True
        return False
    
    #این تابع برای برگشت کتاب توسط کاربر استفاده میشه

    def return_book(self, book: Book) -> None:
        book.return_book()
        self.borrowed_books.remove(book)

    # این تابع اطلاعات کاربر نمایش میده
    def display(self) -> None:
        print(f"{self.name} ({len(self.borrowed_books)} books borrowed)")
# ------------------------------------------
# Library Class
# ------------------------------------------
class Library:
    def __init__(self) -> None:
        self.books: list[Book] = []
        self.users: list[User] = []

    # ---------- Book Management ----------
    
    #این تابع یک کتاب جدید را به لیست کتابخانه اضافه میکند

    def add_book(self, book: Book) -> None:
        self.books.append(book)

    # این تابع یک کتاب از کتابخانه حذف میکند

    def remove_book(self, title: str) -> None:
        for book in self.books:
            if book.title == title:
                self.books.remove(book)
                return

    #این تابع برای پیدا کردن یک کتاب در کتابخانه استفاده میشود

    def search_book(self, title: str) -> Book | None:
        for book in self.books:
            if book.title == title:
                return book
        return None

    # این تابع تمام کتاب های موجود در کتابخانه را نمایش میدهد

    def show_all_books(self) -> None:
        for book in self.books:
            book.display()

    # این تابع کتاب های در دسترس کتابخانه را نمایش میدهد

    def show_available_books(self) -> None:
        for book in self.books:
            if book.available:
                book.display()
    
    # ---------- User Management ----------
    
    #این تابع یک کاربر جدید را به لیست کاربران اضافه میکند

    def add_user(self, user: User) -> None:
        self.users.append(user)
    
    # این تابع برای پیدا کردن یک کاربر در لیست کاربران استفاده میشود

    def search_user(self, name: str) -> User | None:
        for user in self.users:
            if user.name == name:
                return user
            return None
   
   # ---------- Borrow / Return ----------

    #این تابع مسیول پیدا کردن کاربر و کتاب است

    def borrow_book(self, user_name: str, book_title: str) -> None:
        user = self.search_user(user_name)
        book = self.search_book(book_title)
        if user is None:
            print("User not found.")
            return
        if book is None:
            print("Book not found.")
            return
        if user.borrow_book(book):
            print("Book borrowed successfully.")
        else:
            print("Book is already unavailable.")

    #این تابع مسیول پیدا کردن کاربر و کتاب برای عملیات پس دادن است

    def return_book(self, user_name: str, book_title: str) -> None:
        user = self.search_user(user_name)
        book = self.search_book(book_title)
        if user is None:
            print("User not found.")
            return
        if book is None:
            print("Book not found.")
            return
        user.return_book(book)
        print("Book returned successfully.")

    # ---------- File Handling ----------

    #این تابع اطلاغات تمام کتابخانه را دریک فایل ذخبره میکنه

    def save_books(self, filename: str) -> None:
        data = []
        for book in self.books:
            data.append({
                "title": book.title,
                "author": book.author,
                "year": book.year,
                "available": book.available})
            with open(filename, "w") as file:
                json.dump(data, file, indent=4)

    # این تابع اطلاعات را از فایل جیسون میخواند

    def load_books(self, filename: str) -> None:
        with open(filename, "r") as file:
            data = json.load(file)
            self.books = []
            for item in data:
                book = Book(
                    item["title"],
                    item["author"],
                    item["year"])
                book.available = item["available"]
                self.books.append(book)

# ------------------------------------------
# User Interface
# ------------------------------------------
def print_menu():
    print("\n========== Library Management ==========")
    print("Available commands:")
    print("  add_book")
    print("  remove_book")
    print("  search_book")
    print("  show_all_books")
    print("  show_available_books")
    print("  add_user")
    print("  search_user")
    print("  borrow_book")
    print("  return_book")
    print("  save_books")
    print("  load_books")
    print("  exit")


def run(library):
    while True:
        print_menu()

        command = input("\nCommand: ").strip().lower()

        match command:

            case "add_book":
                title = input("Title: ")
                author = input("Author: ")
                year = int(input("Year: "))

                book = Book(title, author, year)
                library.add_book(book)

            case "remove_book":
                title = input("Title: ")
                library.remove_book(title)

            case "search_book":
                title = input("Title: ")
                result = library.search_book(title)

                if result:
                    result.display()
                else:
                    print("Book not found.")

            case "show_all_books":
                library.show_all_books()

            case "show_available_books":
                library.show_available_books()

            case "add_user":
                name = input("User name: ")

                user = User(name)
                library.add_user(user)

            case "search_user":
                name = input("User name: ")
                result = library.search_user(name)

                if result:
                    result.display()
                else:
                    print("User not found.")

            case "borrow_book":
                user_name = input("User name: ")
                book_title = input("Book title: ")

                library.borrow_book(user_name, book_title)

            case "return_book":
                user_name = input("User name: ")
                book_title = input("Book title: ")

                library.return_book(user_name, book_title)

            case "save_books":
                filename = input("File name: ")
                library.save_books(filename)

            case "load_books":
                filename = input("File name: ")
                library.load_books(filename)

            case "exit":
                print("Goodbye!")
                break

            case _:
                print("Unknown command.")


# ------------------------------------------
# Main
# ------------------------------------------
if __name__ == "__main__":

    # Create a library
    library = Library()

    # Create some books
    clean_code = Book("Clean Code", "Robert C. Martin", 2008)
    python_crash_course = Book("Python Crash Course", "Eric Matthes", 2023)

    # Create a user
    alice = User("Alice")

    # Add data to the library
    library.add_book(clean_code)
    library.add_book(python_crash_course)
    library.add_user(alice)

    # Example method calls
    library.show_all_books()
    library.search_book("Clean Code")

    # Start the application
    run(library)
