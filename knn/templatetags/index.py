from django import template
register = template.Library()

@register.filter
def index(list, item):
   return list.index(item)