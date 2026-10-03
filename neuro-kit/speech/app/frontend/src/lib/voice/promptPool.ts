// app/frontend/src/lib/voice/promptPool.ts
export interface Passage { id: string; text: string; }

// Phonetically balanced, neutral, ~35-word passages. Text never leaves the
// client — only the id is transmitted.
export const PASSAGES: Passage[] = [
  { id: 'rainbow_1', text: 'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors that arch across the open sky.' },
  { id: 'walk_1', text: 'A short walk in the morning helps clear the mind before a busy day. The cool air, the quiet streets, and the steady rhythm of each step make the start feel calm and unhurried.' },
  { id: 'library_1', text: 'The old library held thousands of books on tall wooden shelves. Sunlight fell across the reading tables, and the only sound was the soft turning of pages as people studied in comfortable silence.' },
];

export function pickPassage(): Passage {
  return PASSAGES[Math.floor(Math.random() * PASSAGES.length)];
}
