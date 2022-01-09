from flask import Flask
from flask import request
from jinja2 import Environment, PackageLoader
from language_generation.generator import Generator


generator = Generator(initial_category=None, initial_polarity=None)
app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    """
    GET-controller method for start page / main form

    :return: rendered html code for main form with default settings
    """
    return get_html_form(start="", generated_output="", category="Laptops", polarity="positive", max_length=128)


@app.route("/", methods=['POST'])
def generate():
    """
    POST-controller method to process form input and generate a text within main form

    :return: rendered html code for main form with generated text and prefilled settings from request
    """
    start = request.form.get("text")
    category = request.form.get("category")
    polarity = request.form.get("polarity")
    max_len = int(request.form.get("max_length"))
    return get_html_form(start=start, generated_output=generator.generate(start, category, polarity, max_len),
                         category=category, polarity=polarity, max_length=max_len)


def get_html_form(start, generated_output, category, polarity, max_length):
    """
    helper method to pass settings to jinja for parsing the template from templates/template.html

    :param start: beginning of the text to generate (if empty, placeholder instruction is shown)
    :param generated_output: generated text to be shown in main form
    :param category: pre-selected option for category
    :param polarity: pre-selected option for polarity
    :param max_length: pre-selected option for maximum number of tokens
    :return: rendered html code for main form using the parameters
    """
    environment = Environment(loader=PackageLoader('language_generation', 'templates'))
    return environment.get_template("template.html").render(
        category_list=['Laptops', 'Cell Phones', 'Mens Running Shoes', 'Vacuums', 'Plush Figures'],
        generated_output=generated_output,
        prefilled_text=start,
        prefilled_category=category,
        prefilled_polarity=polarity,
        prefilled_max_length=max_length
    )


if __name__ == '__main__':
    app.run()
