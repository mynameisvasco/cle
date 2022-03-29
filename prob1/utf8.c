#include "utf8.h"

int is_vowel(unsigned int c)
{
  int ac_vowels[40] = {
      0x61,
      0x65,
      0x69,
      0x6F,
      0x75,
      0x41,
      0x45,
      0x49,
      0x4F,
      0x55,
      0xC3A1,
      0xC3A0,
      0xC3A2,
      0xC3A3,
      0xC3A9,
      0xC3A8,
      0xC3AA,
      0xC3AD,
      0xC3AC,
      0xC3B3,
      0xC3B2,
      0xC3B4,
      0xC3B5,
      0xC3BA,
      0xC3B9,
      0xC381,
      0xC380,
      0xC382,
      0xC383,
      0xC389,
      0xC388,
      0xC38A,
      0xC38D,
      0xC38C,
      0xC393,
      0xC392,
      0xC394,
      0xC395,
      0xC39A,
      0xC399};

  for (int i = 0; i < 40; i++)
  {
    if (c == ac_vowels[i])
    {
      return 1;
    }
  }

  return 0;
}

int is_consonant(unsigned int c)
{
  int lower_cedilha = 0xC3A7;
  int upper_cedilha = 0xC387;
  return !is_vowel(c) && (((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) || c == lower_cedilha || c == upper_cedilha);
}

int is_separator(unsigned int c)
{
  int dash = 0xE28093;
  int ellipsis = 0xE280A6;
  int doubleQuotationMarkLeft = 0xE2809C;
  int doubleQuotationMarkRight = 0xE2809D;
  return c == '[' || c == ']' || c == '(' || c == ')' || c == '-' || c == '"' || c == 0x20 || c == 0x9 || c == 0xA || c == 0xD || c == '.' || c == '?' || c == ';' || c == ':' || c == ',' || c == '!' || c == dash || c == ellipsis || c == doubleQuotationMarkLeft || c == doubleQuotationMarkRight;
}

int is_number(unsigned int c)
{
  return c >= 48 && c <= 57;
}

int read_u8char(FILE *file)
{
  unsigned char buffer[4] = {0, 0, 0, 0};
  unsigned int c = 0;
  fread(buffer, 1, 1, file);
  c = buffer[0];

  if ((buffer[0] >> 5) == 0b110)
  {
    fread(&buffer[1], 1, 1, file);
    c = (c << 8) | (buffer[1] & 0xff);
  }
  else if ((buffer[0] >> 4) == 0b1110)
  {
    fread(&buffer[1], 1, 2, file);
    c = (c << 8) | (buffer[1] & 0xff);
    c = (c << 8) | (buffer[2] & 0xff);
  }
  else if ((buffer[0] >> 3) == 0b11110)
  {
    fread(&buffer[1], 1, 3, file);
    c = (c << 8) | (buffer[1] & 0xff);
    c = (c << 8) | (buffer[2] & 0xff);
    c = (c << 8) | (buffer[2] & 0xff);
  }

  return c;
}