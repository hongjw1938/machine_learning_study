
# coding: utf-8

# # Appendix: Python Language Essentials

# In[ ]:


from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import *
import pandas
np.set_printoptions(precision=4)


# ## The Python interpreter

# ```
# $ python
# Python 2.7.2 (default, Oct  4 2011, 20:06:09)
# [GCC 4.6.1] on linux2
# Type "help", "copyright", "credits" or "license" for more information.
# >>> a = 5
# >>> print a
# 5
# ```

# In[ ]:


get_ipython().run_cell_magic('writefile', 'hello_world.py', "print 'Hello world'")


# ```
# $ ipython
# Python 2.7.2 |EPD 7.1-2 (64-bit)| (default, Jul  3 2011, 15:17:51)
# Type "copyright", "credits" or "license" for more information.
# 
# IPython 0.12 -- An enhanced Interactive Python.
# ?         -> Introduction and overview of IPython's features.
# %quickref -> Quick reference.
# help      -> Python's own help system.
# object?   -> Details about 'object', use 'object??' for extra details.
# 
# In [1]: %run hello_world.py
# Hello world
# 
# In [2]:
# ```

# ## The Basics

# ### Language Semantics

# #### Indentation, not braces
for x in array:
    if x < pivot:
        less.append(x)
    else:
        greater.append(x)for x in array {
        if x < pivot {
            less.append(x)
        } else {
            greater.append(x)
        }
    }for x in array
    {
      if x < pivot
      {
        less.append(x)
      }
      else
      {
        greater.append(x)
      }
    }a = 5; b = 6; c = 7
# #### Everything is an object

# #### Comments

# In[ ]:


results = []
for line in file_handle:
    # keep the empty lines for now
    # if len(line) == 0:
    #   continue
    results.append(line.replace('foo', 'bar'))


# #### Function and object method calls

# In[ ]:


result = f(x, y, z)
g()


# In[ ]:


obj.some_method(x, y, z)


# In[ ]:


result = f(a, b, c, d=5, e='foo')


# #### Variables and pass-by-reference

# In[ ]:


a = [1, 2, 3]


# In[ ]:


b = a


# In[ ]:


a.append(4)
b


# In[ ]:


def append_element(some_list, element):
    some_list.append(element)


# In[ ]:


data = [1, 2, 3]

append_element(data, 4)

In [4]: data
Out[4]: [1, 2, 3, 4]


# #### Dynamic references, strong types

# In[1]:


from IPython.core.interactiveshell import InteractiveShell


# In[2]:


InteractiveShell.ast_node_interactivity = 'all'  #아래 값을 연속적으로 출력함. 원래 기본값은 last


# In[3]:


a = 5
type(a)
a = 'foo'
type(a)


# In[4]:


'5' + 5


# In[8]:


a = 4.5
b = 2
# String formatting, to be visited later
print('a is %s, b is %s' % (type(a), type(b)))
a / b


# In[9]:


a = 5
isinstance(a, int)


# In[10]:


a = 5; b = 4.5
isinstance(a, (int, float))
isinstance(b, (int, float))


# #### Attributes and methods
In [1]: a = 'foo'

In [2]: a.<Tab>
a.capitalize  a.format      a.isupper     a.rindex      a.strip
a.center      a.index       a.join        a.rjust       a.swapcase
a.count       a.isalnum     a.ljust       a.rpartition  a.title
a.decode      a.isalpha     a.lower       a.rsplit      a.translate
a.encode      a.isdigit     a.lstrip      a.rstrip      a.upper
a.endswith    a.islower     a.partition   a.split       a.zfill
a.expandtabs  a.isspace     a.replace     a.splitlines
a.find        a.istitle     a.rfind       a.startswith
# In[11]:


a = 'foo'
type(a)


# In[12]:


a.split

>>> getattr(a, 'split')
<function split>

# #### "Duck" typing

# In[13]:


dir(a)  #내부의 메서드 추출


# In[ ]:


def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError: # not iterable
        return False


# In[ ]:


isiterable('a string')
isiterable([1, 2, 3])
isiterable(5)

if not isinstance(x, list) and isiterable(x):
    x = list(x)
# #### Imports

# In[ ]:


# some_module.py
PI = 3.14159

def f(x):
    return x + 2

def g(a, b):
    return a + b


# In[14]:


import some_module
result = some_module.f(5)
pi = some_module.PI


# In[ ]:


from some_module import f, g, PI
result = g(5, PI)


# In[ ]:


import some_module as sm
from some_module import PI as pi, g as gf

r1 = sm.f(pi)
r2 = gf(6, pi)


# #### Binary operators and comparisons

# In[ ]:


5 - 7
12 + 21.5
5 <= 2


# In[15]:


a = [1, 2, 3]
b = a
# Note, the list function always creates a new list
c = list(a)
a is b
a is not c


# In[ ]:


a == c


# In[ ]:


a = None
a is None


# #### Strictness versus laziness

# In[16]:


a = b = c = 5
d = a + b * c


# In[17]:


print(a, b, c, d)


# #### Mutable and immutable objects

# In[18]:


a_list = ['foo', 2, [4, 5]]
a_list[2] = (3, 4)
a_list


# In[ ]:


a_tuple = (3, 5, (4, 5))
a_tuple[1] = 'four'


# ### Scalar Types

# #### Numeric types

# In[19]:


ival = 17239871
ival ** 6


# In[20]:


fval = 7.243
fval2 = 6.78e-5


# In[21]:


3 / 2


# In[24]:


from __future__ import division


# In[25]:


3 / float(2)


# In[26]:


3 // 2


# In[27]:


cval = 1 + 2j
cval * (1 - 2j)


# #### Strings

# In[ ]:


a = 'one way of writing a string'
b = "another way"


# In[ ]:


c = """
This is a longer string that
spans multiple lines
"""


# In[30]:


a = 'this is a string'
a[10] = 'f'    #str class is immutable


# In[29]:


b = a.replace('string', 'longer string')
b


# In[ ]:


a = 5.6
s = str(a)
s


# In[ ]:


s = 'python'
list(s)
s[:3]


# In[ ]:


s = '12\\34'
print s


# In[ ]:


s = r'this\has\no\special\characters'
s


# In[ ]:


a = 'this is the first half '
b = 'and this is the second half'
a + b


# In[31]:


template = '%.2f %s are worth $%d'


# In[32]:


template % (4.5560, 'Argentine Pesos', 1)


# #### Booleans

# In[ ]:


True and True
False or True


# In[ ]:


a = [1, 2, 3]
if a:
    print 'I found something!'

b = []
if not b:
    print 'Empty!'


# In[ ]:


bool([]), bool([1, 2, 3])
bool('Hello world!'), bool('')
bool(0), bool(1)


# #### Type casting

# In[ ]:


s = '3.14159'
fval = float(s)
type(fval)
int(fval)
bool(fval)
bool(0)


# #### None

# In[33]:


a = None
a is None
b = 5
b is not None


# In[44]:


def add_and_maybe_multiply(a, b, c=None):
    result = a + b
    print(result)
    if c is not None:
        result = result * c

    return result


# In[46]:


add_and_maybe_multiply(4,5,1)


# #### Dates and times

# In[39]:


from datetime import datetime, date, time
dt = datetime(2011, 10, 29, 20, 30, 21)
dt.day
dt.minute


# In[40]:


dt.date()  #date클래스의 객체
dt.time()  #time클래스의 객체


# In[41]:


dt.strftime('%m/%d/%Y %H:%M')  #시간을 출력시, 포맷을 정함


# In[47]:


datetime.strptime('20091031', '%Y%m%d')


# In[48]:


dt.replace(minute=0, second=0)


# In[49]:


dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt
delta
type(delta)


# In[50]:


dt
dt + delta


# ### Control Flow

# #### If, elif, and else

# In[ ]:


if x < 0:
    print 'It's negative'


# In[ ]:


if x < 0:
    print 'It's negative'
elif x == 0:
    print 'Equal to zero'
elif 0 < x < 5:
    print 'Positive but smaller than 5'
else:
    print 'Positive and larger than 5'


# In[ ]:


a = 5; b = 7
c = 8; d = 4
if a < b or c > d:   #shortcut evaluation
    print 'Made it'


# #### For loops

# In[51]:


for value in collection:
    # do something with value


# In[56]:


sequence = [1, 2, None, 4, None, 5]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value    


# In[57]:


total


# In[53]:


sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0
for value in sequence:
    if value == 5:
        break
    total_until_5 += value


# In[54]:


for a, b, c in iterator:
    # do something


# In[67]:


values = [(1,2), (3,4), (10,20)]


# In[68]:


for value in values:
    print(value)


# In[69]:


for (k,v) in values:
    print('key: %d, value: %d' % (k, v))


# While loops

# In[ ]:


x = 256
total = 0
while x > 0:
    if total > 500:
        break
    total += x
    x = x // 2


# #### pass

# In[ ]:


if x < 0:
    print 'negative!'
elif x == 0:
    # TODO: put something smart here
    pass
else:
    print 'positive!'


# In[ ]:


def f(x, y, z):
    # TODO: implement this function!
    pass


# #### Exception handling

# In[70]:


float('1.2345')
float('something')


# In[71]:


def attempt_float(x):
    try:
        return float(x)
    except:
        return x


# In[72]:


attempt_float('1.2345')
attempt_float('something')


# In[73]:


float((1, 2))


# In[76]:


def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x


# In[79]:


attempt_float((1, 2))


# In[78]:


def attempt_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


# In[81]:




try:
    f = open(path, 'w')
    write_to_file(f)
except NameError:
    print("경로가 잘못되었거나 지정되지 않았습니다.")
finally:
    f.close()


# In[ ]:


f = open(path, 'w')

try:
    write_to_file(f)
except:
    print 'Failed'
else:
    print 'Succeeded'
finally:
    f.close()


# #### range and xrange

# In[84]:


range(10)  #python3에서는 객체를 리턴함. 직접 반환하지 않고 객체로 준비만 시킴.


# In[83]:


range(0, 20, 2)


# In[87]:


seq = [1, 2, 3, 4]
for i in range(len(seq)):
    val = seq[i]
    val


# In[ ]:


sum = 0
for i in xrange(10000):
    # % is the modulo operator
    if i % 3 == 0 or i % 5 == 0:
        sum += i


# #### Ternary Expressions

# In[90]:


x = 5
value = 'Non-negative' if x >= 0 else 'Negative'  #삼단표현방식 x>=0이라면 'Non-negative'이며 아니면 'Negative'라는 의미
value


# ## Data structures and sequences

# ### Tuple

# In[ ]:


tup = 4, 5, 6
tup


# In[ ]:


nested_tup = (4, 5, 6), (7, 8)
nested_tup


# In[ ]:


tuple([4, 0, 2])
tup = tuple('string')
tup


# In[ ]:


tup[0]


# In[ ]:


tup = tuple(['foo', [1, 2], True])
tup[2] = False

# however
tup[1].append(3)
tup


# In[ ]:


(4, None, 'foo') + (6, 0) + ('bar',)


# In[ ]:


('foo', 'bar') * 4


# #### Unpacking tuples

# In[ ]:


tup = (4, 5, 6)
a, b, c = tup
b


# In[91]:


tup = 4, 5, (6, 7)
a, b, (c, d) = tup
d

tmp = a
a = b
b = tmpb, a = a, b
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
    pass
# #### Tuple methods

# In[92]:


a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)


# In[93]:


dir(a)


# ### List

# In[94]:


a_list = [2, 3, 7, None]

tup = ('foo', 'bar', 'baz')
b_list = list(tup)
b_list
b_list[1] = 'peekaboo'
b_list


# #### Adding and removing elements

# In[95]:


b_list.append('dwarf')
b_list


# In[96]:


b_list.insert(1, 'red')
b_list


# In[97]:


b_list.pop(2)
b_list


# In[ ]:


b_list.append('foo')
b_list.remove('foo')
b_list


# In[ ]:


'dwarf' in b_list


# #### Concatenating and combining lists

# In[ ]:


[4, None, 'foo'] + [7, 8, (2, 3)]


# In[ ]:


x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
x


# In[ ]:


everything = []
for chunk in list_of_lists:
    everything.extend(chunk)


# In[ ]:


everything = []
for chunk in list_of_lists:
    everything = everything + chunk


# #### Sorting

# In[98]:


a = [7, 2, 5, 1, 3]
a.sort()
a


# In[105]:


a.reverse()
a


# In[ ]:


b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
b


# #### Binary search and maintaining a sorted list

# In[110]:


import bisect
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 4)
bisect.bisect(c, 2)
bisect.bisect(c, 5)
bisect.insort(c, 6)
c


# #### Slicing

# In[ ]:


seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:5]


# In[ ]:


seq[3:4] = [6, 3]
seq


# In[ ]:


seq[:5]
seq[3:]


# In[ ]:


seq[-4:]
seq[-6:-2]


# In[ ]:


seq[::2]


# In[ ]:


seq[::-1]


# ### Built-in Sequence Functions

# #### enumerate

# In[ ]:



i = 0
for value in collection:
   # do something with value
   i += 1


# In[ ]:


for i, value in enumerate(collection):
   # do something with value


# In[ ]:


some_list = ['foo', 'bar', 'baz']
mapping = dict((v, i) for i, v in enumerate(some_list))
mapping


# #### sorted

# In[ ]:


sorted([7, 1, 2, 6, 0, 3, 2])
sorted('horse race')


# In[111]:


x = [7, 3, 2, 0, 8]
sorted(x)  #sort와의 차이점은 결과에서 보듯이, 새로운 정렬된 자료형을 반환하고 기존 내용의 정렬상태는 그대로 유지함.
x


# In[ ]:


sorted(set('this is just some string')) #unique한 대표값 하나만 남김. 중복값 1개만.


# #### zip

# In[ ]:


seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zip(seq1, seq2)


# In[ ]:


seq3 = [False, True]
zip(seq1, seq2, seq3)


# In[ ]:


for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('%d: %s, %s' % (i, a, b))


# In[ ]:


pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),
            ('Schilling', 'Curt')]
first_names, last_names = zip(*pitchers)
first_names
last_names


# In[ ]:


zip(seq[0], seq[1], ..., seq[len(seq) - 1])


# #### reversed

# In[ ]:


list(reversed(range(10)))


# ### Dict

# In[ ]:


empty_dict = {}
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
d1


# In[ ]:


d1[7] = 'an integer'
d1
d1['b']


# In[ ]:


'b' in d1


# In[ ]:


d1[5] = 'some value'
d1['dummy'] = 'another value'
del d1[5]
ret = d1.pop('dummy')
ret


# In[ ]:


d1.keys()
d1.values()


# In[ ]:


d1.update({'b' : 'foo', 'c' : 12})
d1


# #### Creating dicts from sequences

# In[ ]:


mapping = {}
for key, value in zip(key_list, value_list):
    mapping[key] = value


# In[ ]:


mapping = dict(zip(range(5), reversed(range(5))))
mapping


# #### Default values

# In[ ]:


if key in some_dict:
    value = some_dict[key]
else:
    value = default_value


# In[ ]:


value = some_dict.get(key, default_value)


# In[ ]:


words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}

for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)

by_letter

by_letter.setdefault(letter, []).append(word)
# In[ ]:


from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)


# In[ ]:


counts = defaultdict(lambda: 4)


# #### Valid dict key types

# In[ ]:


hash('string')
hash((1, 2, (2, 3)))
hash((1, 2, [2, 3])) # fails because lists are mutable


# In[ ]:


d = {}
d[tuple([1, 2, 3])] = 5
d


# ### Set

# In[ ]:


set([2, 2, 2, 1, 3, 3])
{2, 2, 2, 1, 3, 3}


# In[ ]:


a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
a | b  # union (or)
a & b  # intersection (and)
a - b  # difference
a ^ b  # symmetric difference (xor)


# In[ ]:


a_set = {1, 2, 3, 4, 5}
{1, 2, 3}.issubset(a_set)
a_set.issuperset({1, 2, 3})


# In[ ]:


{1, 2, 3} == {3, 2, 1}


# ### List, set, and dict comprehensions

# In[ ]:


strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]


# In[ ]:


unique_lengths = {len(x) for x in strings}
unique_lengths


# In[ ]:


loc_mapping = {val : index for index, val in enumerate(strings)}
loc_mapping

loc_mapping = dict((val, idx) for idx, val in enumerate(strings)}
# #### Nested list comprehensions

# In[ ]:


all_data = [['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Steven', 'Joe'],
            ['Susie', 'Casey', 'Jill', 'Ana', 'Eva', 'Jennifer', 'Stephanie']]


# In[ ]:


names_of_interest = []
for names in all_data:
    enough_es = [name for name in names if name.count('e') > 2]
    names_of_interest.extend(enough_es)


# In[ ]:


result = [name for names in all_data for name in names
          if name.count('e') >= 2]
result


# In[ ]:


some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened


# In[ ]:


flattened = []

for tup in some_tuples:
    for x in tup:
        flattened.append(x)


# In[ ]:


[[x for x in tup] for tup in some_tuples]


# ## Functions

# In[ ]:


def my_function(x, y, z=1.5):
    if z > 1:
        return z * (x + y)
    else:
        return z / (x + y)


# In[ ]:


my_function(5, 6, z=0.7)
my_function(3.14, 7, 3.5)


# ### Namespaces, scope, and local functions

# In[ ]:


def func():
    a = []
    for i in range(5):
        a.append(i)


# In[ ]:


a = []
def func():
    for i in range(5):
        a.append(i)


# In[ ]:


a = None
def bind_a_variable():
    global a
    a = []
bind_a_variable()
print a


# In[ ]:


def outer_function(x, y, z):
    def inner_function(a, b, c):
        pass
    pass


# ### Returning multiple values

# In[ ]:


def f():
    a = 5
    b = 6
    c = 7
    return a, b, c

a, b, c = f()


# In[ ]:


return_value = f()


# In[ ]:


def f():
    a = 5
    b = 6
    c = 7
    return {'a' : a, 'b' : b, 'c' : c}


# ### Functions are objects

# In[ ]:



states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda',
          'south   carolina##', 'West virginia?']


# In[ ]:


import re  # Regular expression module

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value) # remove punctuation
        value = value.title()
        result.append(value)
    return result


# In[ ]:


clean_strings(states)
Out[15]:
['Alabama',
 'Georgia',
 'Georgia',
 'Georgia',
 'Florida',
 'South Carolina',
 'West Virginia']


# In[ ]:


def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result


# In[ ]:


clean_strings(states, clean_ops)
Out[22]:
['Alabama',
 'Georgia',
 'Georgia',
 'Georgia',
 'Florida',
 'South Carolina',
 'West Virginia']


# In[ ]:


map(remove_punctuation, states)
Out[23]:
['   Alabama ',
 'Georgia',
 'Georgia',
 'georgia',
 'FlOrIda',
 'south   carolina',
 'West virginia']


# ### Anonymous (lambda) functions

# In[ ]:


def short_function(x):
    return x * 2

equiv_anon = lambda x: x * 2


# In[ ]:


def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

ints = [4, 0, 1, 5, 6]
apply_to_list(ints, lambda x: x * 2)


# In[ ]:


strings = ['foo', 'card', 'bar', 'aaaa', 'abab']


# In[ ]:


strings.sort(key=lambda x: len(set(list(x))))
strings


# ### Closures: functions that return functions

# In[ ]:


def make_closure(a):
    def closure():
        print('I know the secret: %d' % a)
    return closure

closure = make_closure(5)


# In[ ]:


def make_watcher():
    have_seen = {}

    def has_been_seen(x):
        if x in have_seen:
            return True
        else:
            have_seen[x] = True
            return False

    return has_been_seen


# In[ ]:


watcher = make_watcher()
vals = [5, 6, 1, 5, 1, 6, 3, 5]
[watcher(x) for x in vals]

def make_counter():
    count = [0]
    def counter():
        # increment and return the current count
        count[0] += 1
        return count[0]
    return counter

counter = make_counter()
# In[ ]:


def format_and_pad(template, space):
    def formatter(x):
        return (template % x).rjust(space)

    return formatter


# In[ ]:


fmt = format_and_pad('%.4f', 15)
fmt(1.756)


# ### Extended call syntax with *args, **kwargs

# In[ ]:


a, b, c = args
d = kwargs.get('d', d_default_value)
e = kwargs.get('e', e_default_value)


# In[ ]:


def say_hello_then_call_f(f, *args, **kwargs):
    print 'args is', args
    print 'kwargs is', kwargs
    print("Hello! Now I'm going to call %s" % f)
    return f(*args, **kwargs)

def g(x, y, z=1):
    return (x + y) / z


# In[ ]:


 say_hello_then_call_f(g, 1, 2, z=5.)
args is (1, 2)
kwargs is {'z': 5.0}
Hello! Now I'm going to call <function g at 0x2dd5cf8>
Out[8]: 0.6


# ### Currying: partial argument application

# In[ ]:


def add_numbers(x, y):
    return x + y


# In[ ]:


add_five = lambda y: add_numbers(5, y)


# In[ ]:


from functools import partial
add_five = partial(add_numbers, 5)


# In[ ]:


# compute 60-day moving average of time series x
ma60 = lambda x: pandas.rolling_mean(x, 60)

# Take the 60-day moving average of of all time series in data
data.apply(ma60)


# ### Generators

# In[ ]:


some_dict = {'a': 1, 'b': 2, 'c': 3}
for key in some_dict:
    print key,


# In[ ]:


dict_iterator = iter(some_dict)
dict_iterator


# In[ ]:


list(dict_iterator)


# In[ ]:


def squares(n=10):
    for i in xrange(1, n + 1):
        print 'Generating squares from 1 to %d' % (n ** 2)
        yield i ** 2


# In[ ]:


gen = squares()

gen
Out[3]: <generator object squares at 0x34c8280>


# In[ ]:


for x in gen:
    print x,

Generating squares from 0 to 100
1 4 9 16 25 36 49 64 81 100


# In[ ]:


def make_change(amount, coins=[1, 5, 10, 25], hand=None):
    hand = [] if hand is None else hand
    if amount == 0:
        yield hand
    for coin in coins:
        # ensures we don't give too much change, and combinations are unique
        if coin > amount or (len(hand) > 0 and hand[-1] < coin):
            continue

        for result in make_change(amount - coin, coins=coins,
                                  hand=hand + [coin]):
            yield result


# In[ ]:


for way in make_change(100, coins=[10, 25, 50]):
    print way
len(list(make_change(100)))


# #### Generator expresssions

# In[ ]:


gen = (x ** 2 for x in xrange(100))
gen


# In[ ]:


def _make_gen():
    for x in xrange(100):
        yield x ** 2
gen = _make_gen()


# In[ ]:


sum(x ** 2 for x in xrange(100))
dict((i, i **2) for i in xrange(5))


# #### itertools module

# In[ ]:


import itertools
first_letter = lambda x: x[0]

names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']

for letter, names in itertools.groupby(names, first_letter):
    print letter, list(names) # names is a generator


# ## Files and the operating system

# In[ ]:


path = 'ch13/segismundo.txt'
f = open(path)


# In[ ]:


for line in f:
    pass


# In[ ]:


lines = [x.rstrip() for x in open(path)]
lines


# In[ ]:


with open('tmp.txt', 'w') as handle:
    handle.writelines(x for x in open(path) if len(x) > 1)

open('tmp.txt').readlines()


# In[ ]:


os.remove('tmp.txt')

