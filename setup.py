from setuptools import setup

setup(name='funniest',
      version='0.1',
      description='The funniest joke in the world',
      url='http://github.com/storborg/funniest',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['expense_classifier'],
      scripts='expense_classifier/predict_ofx',
      zip_safe=False)
