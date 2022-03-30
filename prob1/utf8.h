#include <stdio.h>

int is_vowel(unsigned int c);

int is_consonant(unsigned int c);

int is_separator(unsigned int c);

int is_number(unsigned int c);

int read_u8char(FILE *file, int *read_bytes);
int read_u8char_buffer(unsigned char *buffer, int *read_bytes);