def func(arg1="foo", arg_a= "bar", firstarg=1):
   print(arg1, arg_a, firstarg)

arguments_dictionary = {
  'arg1': "foo",
  'arg_a': "bar",
  'firstarg':42
   }

func(**arguments_dictionary)

