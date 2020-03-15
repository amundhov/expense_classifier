from setuptools import setup

setup(name='expense_classifier',
      version='0.1',
      description='The expense classifier to keep personal finances up to date. While having fun.',
      url='http://github.com/amundhov/expense_classifier',
      author='Amund Hov',
      author_email='amund.hov@gmail.com',
      license='MIT',
      packages=['expense_classifier'],
      scripts='expense_classifier/predict_ofx',
      zip_safe=False)
