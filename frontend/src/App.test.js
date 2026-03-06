import { render, screen } from '@testing-library/react';
import App from './App';

test('renders detector heading', () => {
  render(<App />);
  const heading = screen.getByText(/choose your detection mode/i);
  expect(heading).toBeInTheDocument();
});
