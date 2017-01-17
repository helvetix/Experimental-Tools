import urllib2, urllib, json
clientId="dj0yJmk9UHlVU3N0b3c5eXRqJmQ9WVdrOU5VdG1kVVY0TkdzbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD0zYQ--"
clientSecret="f57ea6ad7537009773acfd92029d5d6126736e77"

baseurl = "https://query.yahooapis.com/v1/public/yql?"
yql_query = "select wind from weather.forecast where woeid=2460286"
yql_url = baseurl + urllib.urlencode({'q':yql_query}) + "&format=json"
result = urllib2.urlopen(yql_url).read()
data = json.loads(result)

print data['query']['results']

