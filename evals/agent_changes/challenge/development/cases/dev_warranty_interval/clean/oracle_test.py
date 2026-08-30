from warranty import parse_warranty


try:
    parse_warranty({"starts_on": "2027-04-01", "ends_on": "2027-03-31"})
except ValueError:
    pass
else:
    raise AssertionError("reversed interval was accepted")
